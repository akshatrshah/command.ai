package ai.command;

import java.io.*;
import java.lang.ProcessHandle;
import java.nio.charset.StandardCharsets;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Command.ai — Terminal observability + privilege enforcement + ML risk assessment.
 *
 * Usage:
 *   java -jar commandai.jar [--commands path/to/commands.csv] [--user username] [--tier 0|1|2]
 *
 * Privilege tiers (match your CSV privilege_level column directly):
 *   0 = Safe / Read-only   → LOW  risk — execute freely
 *   1 = Normal ops         → MEDIUM risk — warn + confirm
 *   2 = Destructive/Root   → HIGH risk  — blocked
 *
 * A user at tier T can run commands with privilege_level <= T.
 */
public class CommandAI {

    // ── Constants ────────────────────────────────────────────────────────────

    private static final String  VERSION     = "1.0.0";
    private static final String  LOG_DIR     = System.getProperty("user.home") + "/.command_ai/logs";
    private static final String  SESSION_DIR = System.getProperty("user.home") + "/.command_ai/sessions";
    private static final String  ML_SCRIPT   = "ml_engine.py";
    private static final int     MAX_TOKENS  = 8; // max command tokens to try for prefix match

    private static final DateTimeFormatter TS       = DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");
    private static final DateTimeFormatter HUMAN_TS = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

    // ANSI colors
    private static final String RESET  = "\033[0m";
    private static final String RED    = "\033[0;31m";
    private static final String YELLOW = "\033[0;33m";
    private static final String GREEN  = "\033[0;32m";
    private static final String CYAN   = "\033[0;36m";
    private static final String BOLD   = "\033[1m";
    private static final String DIM    = "\033[2m";

    // ── State ────────────────────────────────────────────────────────────────

    private final String    sessionId;
    private final Path      logFile;
    private final Path      sessionIndexFile;
    private final String    username;
    private final int       userTier;
    private String          group; // optional tag, null if ungrouped — mutable so dashboard edits persist
    private final String    startedAt;

    private final List<LogEntry>          sessionEntries = new ArrayList<>();
    private final Map<String, CommandDef> knownCommands  = new LinkedHashMap<>();

    private final MlEngine       ml;
    private final BufferedReader console = new BufferedReader(new InputStreamReader(System.in));
    private volatile boolean     isShuttingDown = false;

    // ── Entry Point ──────────────────────────────────────────────────────────

    public static void main(String[] args) throws Exception {
        String csvPath  = null;
        String username = null;
        String group    = null;
        int    tier     = -1;
        boolean dashboard = false;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "--commands"  -> { if (i + 1 < args.length) csvPath  = args[++i]; }
                case "--user"      -> { if (i + 1 < args.length) username = args[++i]; }
                case "--tier"      -> { if (i + 1 < args.length) tier     = Integer.parseInt(args[++i]); }
                case "--group"     -> { if (i + 1 < args.length) group    = args[++i]; }
                case "--dashboard" -> dashboard = true;
                case "--help", "-h" -> { printHelp(); return; }
            }
        }

        if (dashboard) { Dashboard.run(); return; }

        CommandAI app = new CommandAI(username, tier, group);
        if (csvPath != null) app.loadCommandsFromCsv(csvPath);
        app.start();
    }

    // ── Constructor ──────────────────────────────────────────────────────────

    public CommandAI(String username, int tier, String group) throws IOException {
        long   pid       = ProcessHandle.current().pid();
        String timestamp = LocalDateTime.now().format(TS);
        this.sessionId   = timestamp + "_" + pid;
        this.group       = (group != null && !group.isBlank()) ? group.trim() : null;
        this.startedAt   = LocalDateTime.now().format(HUMAN_TS);

        Files.createDirectories(Paths.get(LOG_DIR));
        Files.createDirectories(Paths.get(SESSION_DIR));
        this.logFile          = Paths.get(LOG_DIR, "session_" + sessionId + ".jsonl");
        this.sessionIndexFile = Paths.get(SESSION_DIR, sessionId + ".json");
        this.username = (username != null) ? username : System.getProperty("user.name", "unknown");
        this.userTier = (tier >= 0 && tier <= 2) ? tier : promptTier();

        this.ml = new MlEngine();
        this.ml.start();
    }

    // ── Session Loop ─────────────────────────────────────────────────────────

    public void start() throws Exception {
        printBanner();
        writeSessionMeta("SESSION_START", null);
        writeSessionIndex(null); // initial index entry, marks session as live
        Runtime.getRuntime().addShutdownHook(new Thread(this::onShutdown));

        String prompt = buildPrompt();
        System.out.print(prompt);
        String line;
        while ((line = console.readLine()) != null) {
            line = line.trim();
            if (line.isEmpty()) { System.out.print(prompt); continue; }
            if (handleBuiltin(line)) { System.out.print(prompt); continue; }
            processCommand(line);
            writeSessionIndex(null); // keep index fresh as commands run
            System.out.print(prompt);
        }
    }

    // ── Command Processing ───────────────────────────────────────────────────

    private void processCommand(String raw) {
        String     ts      = LocalDateTime.now().format(HUMAN_TS);
        CommandDef known   = resolveKnownCommand(raw);
        int        cmdTier = (known != null) ? known.privilegeLevel() : inferTier(raw);
        String     desc    = (known != null) ? known.description() : "Unknown command";

        // Privilege check first
        if (cmdTier > userTier) {
            printError(String.format(
                "PRIVILEGE DENIED: '%s' requires level %d, your session is level %d.",
                raw.split("\\s+")[0], cmdTier, userTier));
            log(raw, "BLOCKED_PRIVILEGE", tierToRisk(cmdTier), ts, desc, -1, null);
            return;
        }

        // Risk determination:
        // 1. Known command (in CSV): tier maps directly to risk — fast, accurate
        // 2. Unknown command: check hard rules first (fork bombs, rm -rf, etc.)
        //    If no rule match → LOW. We don't let the ML model alone block execution
        //    since it was trained on your CSV and will misclassify anything outside it.
        String risk;
        if (known != null) {
            risk = tierToRisk(cmdTier);
        } else {
            String ruleOverride = MlEngine.ruleBasedOverride(raw);
            risk = (ruleOverride != null) ? ruleOverride : "LOW";
        }

        switch (risk) {
            case "HIGH" -> {
                printError("⛔  HIGH RISK — Command blocked: " + raw);
                printDim("    " + desc);
                log(raw, "BLOCKED_RISK", risk, ts, desc, -1, null);
            }
            default -> {
                // LOW and MEDIUM both execute — no prompt, no slowdown
                // Risk level is still logged for audit purposes
                executeAndLog(raw, risk, ts, desc);
            }
        }
    }

    private void executeAndLog(String cmd, String risk, String ts, String desc) {
        long start   = System.currentTimeMillis();
        int  exit    = executeShell(cmd);
        long elapsed = System.currentTimeMillis() - start;
        log(cmd, exit == 0 ? "OK" : "ERROR", risk, ts, desc, exit, elapsed + "ms");
    }

    private int executeShell(String command) {
        try {
            Process p = new ProcessBuilder("bash", "-c", command).inheritIO().start();
            return p.waitFor();
        } catch (Exception e) {
            printError("Execution error: " + e.getMessage());
            return -1;
        }
    }

    // ── Built-in Commands ────────────────────────────────────────────────────

    private boolean handleBuiltin(String line) throws IOException {
        return switch (line.split("\\s+")[0]) {
            case "exit", "quit" -> { shutdown(); yield true; }
            case ":help"        -> { printHelp(); yield true; }
            case ":summary"     -> { printSummary(); yield true; }
            case ":history"     -> { printHistory(); yield true; }
            case ":status"      -> { printStatus(); yield true; }
            case ":logs"        -> { openLogs(); yield true; }
            case ":reload"      -> {
                String[] p = line.split("\\s+", 2);
                if (p.length > 1) loadCommandsFromCsv(p[1].trim());
                else printError("Usage: :reload <path/to/commands.csv>");
                yield true;
            }
            default -> false;
        };
    }

    // ── Logging ──────────────────────────────────────────────────────────────

    private void log(String cmd, String status, String risk, String ts,
                     String desc, int exit, String elapsed) {
        LogEntry entry = new LogEntry(sessionId, username, userTier, ts,
                                      cmd, status, risk, desc, exit, elapsed);
        sessionEntries.add(entry);
        // Write JSONL (machine readable, for parsing/dashboards)
        appendToLog(entry.toJsonLine());
        // Write human-readable line immediately after
        appendToLog(entry.toHumanLine());
    }

    private void appendToLog(String line) {
        try (BufferedWriter w = Files.newBufferedWriter(logFile, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.APPEND)) {
            w.write(line); w.newLine();
        } catch (IOException ex) {
            printError("Log write failed: " + ex.getMessage());
        }
    }

    private void writeSessionMeta(String type, String aiSummary) {
        String ts = LocalDateTime.now().format(HUMAN_TS);
        String line;
        if ("SESSION_START".equals(type)) {
            line = String.format(
                "{\"type\":\"SESSION_START\",\"session\":\"%s\",\"user\":\"%s\"," +
                "\"tier\":%d,\"pid\":%d,\"started\":\"%s\"}",
                sessionId, username, userTier,
                ProcessHandle.current().pid(), ts);
            appendToLog(line);
            appendToLog(String.format(
                "# ═══ SESSION START ══════════════════════════════════════",
                sessionId));
            appendToLog(String.format(
                "#   User    : %s  (tier %d — %s)", username, userTier, tierName(userTier)));
            appendToLog(String.format("#   Session : %s", sessionId));
            appendToLog(String.format("#   Started : %s", ts));
            appendToLog(String.format(
                "# ─────────────────────────────────────────────────────────"));
        } else {
            // SESSION_END
            long total   = sessionEntries.size();
            long blocked = sessionEntries.stream().filter(e -> e.status().startsWith("BLOCKED")).count();
            long high    = sessionEntries.stream().filter(e -> e.risk().equals("HIGH")).count();
            line = String.format(
                "{\"type\":\"SESSION_END\",\"session\":\"%s\",\"ended\":\"%s\"," +
                "\"total_commands\":%d,\"blocked\":%d,\"high_risk\":%d}",
                sessionId, ts, total, blocked, high);
            appendToLog(line);
            appendToLog("# ─────────────────────────────────────────────────────────");
            appendToLog(String.format("#   Commands : %d total, %d blocked, %d high-risk", total, blocked, high));
            if (aiSummary != null) appendToLog("#   Summary  : " + aiSummary);
            appendToLog(String.format("#   Ended    : %s", ts));
            appendToLog("# ═══ SESSION END ════════════════════════════════════════");
        }
    }

    /**
     * Writes a lightweight session index file to ~/.command_ai/sessions/<id>.json
     * used by the --dashboard mode to list past and live sessions without
     * needing to parse full JSONL logs.
     */
    /**
     * Reads this session's own index file and adopts its "group" value if
     * present, so a group change made from the dashboard while this session
     * is still live isn't immediately overwritten by the next writeSessionIndex().
     * Safe no-op on the very first call, since the index file won't exist yet.
     */
    private void syncGroupFromDisk() {
        if (!Files.exists(sessionIndexFile)) return;
        try {
            String json = Files.readString(sessionIndexFile, StandardCharsets.UTF_8);
            String diskGroup = readJsonStringField(json, "group");
            // diskGroup is null both when the field is JSON null and when it's absent —
            // either way that means "no group", which is a valid value to adopt.
            this.group = diskGroup;
        } catch (IOException ignored) {
            // If we can't read it, just keep whatever group we already have in memory.
        }
    }

    /** Minimal JSON string-field reader, mirrors the one used by Dashboard. */
    private static String readJsonStringField(String json, String key) {
        for (String search : List.of("\"" + key + "\":\"", "\"" + key + "\": \"")) {
            int s = json.indexOf(search);
            if (s >= 0) {
                s += search.length();
                StringBuilder sb = new StringBuilder();
                boolean escaped = false;
                for (int i = s; i < json.length(); i++) {
                    char c = json.charAt(i);
                    if (escaped) { sb.append(c); escaped = false; }
                    else if (c == '\\') escaped = true;
                    else if (c == '"') break;
                    else sb.append(c);
                }
                return sb.toString();
            }
        }
        return null; // covers JSON null and missing key — both mean "no group"
    }

    private void writeSessionIndex(String aiSummary) {
        // If the dashboard changed our group while we were live, pick that up
        // instead of stomping it with our in-memory --group value.
        syncGroupFromDisk();

        long total   = sessionEntries.size();
        long blocked = sessionEntries.stream().filter(e -> e.status().startsWith("BLOCKED")).count();
        long high    = sessionEntries.stream().filter(e -> e.risk().equals("HIGH")).count();
        long pid     = ProcessHandle.current().pid();
        boolean alive = !isShuttingDown;

        String json = String.format(
            "{\"session\":\"%s\",\"user\":\"%s\",\"tier\":%d,\"group\":%s," +
            "\"pid\":%d,\"alive\":%b,\"started\":\"%s\",\"updated\":\"%s\"," +
            "\"total_commands\":%d,\"blocked\":%d,\"high_risk\":%d," +
            "\"log_file\":\"%s\",\"summary\":%s}",
            sessionId, esc(username), userTier,
            group != null ? "\"" + esc(group) + "\"" : "null",
            pid, alive, startedAt, LocalDateTime.now().format(HUMAN_TS),
            total, blocked, high,
            esc(logFile.toString()),
            aiSummary != null ? "\"" + esc(aiSummary) + "\"" : "null"
        );

        try {
            Files.writeString(sessionIndexFile, json, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        } catch (IOException ignored) {
            // Non-fatal — dashboard just won't see this session
        }
    }

    private static String esc(String s) {
        if (s == null) return "";
        return s.replace("\\","\\\\").replace("\"","\\\"")
                .replace("\n","\\n").replace("\r","\\r");
    }

    // ── Summary / History ────────────────────────────────────────────────────

    private void printSummary() {
        if (sessionEntries.isEmpty()) { printInfo("No commands yet."); return; }

        long total     = sessionEntries.size();
        long blocked   = sessionEntries.stream().filter(e -> e.status().startsWith("BLOCKED")).count();
        long errors    = sessionEntries.stream().filter(e -> e.status().equals("ERROR")).count();
        long ok        = sessionEntries.stream().filter(e -> e.status().equals("OK")).count();
        long high      = sessionEntries.stream().filter(e -> e.risk().equals("HIGH")).count();
        long medium    = sessionEntries.stream().filter(e -> e.risk().equals("MEDIUM")
                             && !e.status().startsWith("BLOCKED")).count();

        System.out.println();
        System.out.println(CYAN + BOLD + "── Session Summary ─────────────────────────────────" + RESET);
        System.out.printf("  Commands      : %d total  (%s%d ok%s  %s%d errors%s  %s%d blocked%s)%n",
            total,
            GREEN,                    ok,      RESET,
            errors  > 0 ? YELLOW : DIM, errors,  RESET,
            blocked > 0 ? RED    : DIM, blocked, RESET);
        System.out.printf("  Risk profile  : %s%d HIGH blocked%s  /  %s%d MEDIUM executed%s%n",
            high   > 0 ? RED    : DIM, high,   RESET,
            medium > 0 ? YELLOW : DIM, medium, RESET);
        System.out.println();
        System.out.print(CYAN + "Generating AI summary... " + RESET);
        System.out.flush();

        // Build session text for ML summarizer
        StringBuilder sb = new StringBuilder();
        for (LogEntry e : sessionEntries)
            sb.append(e.status()).append(": ").append(e.command())
              .append(" [").append(e.risk()).append("]; ");

        String summary = ml.summarize(sb.toString());
        System.out.println();
        System.out.println(BOLD + "  " + summary + RESET);
        System.out.println(CYAN + "────────────────────────────────────────────────────" + RESET);
        System.out.println();
    }

    private void printHistory() {
        if (sessionEntries.isEmpty()) { printInfo("No history yet."); return; }
        System.out.println();
        System.out.println(CYAN + BOLD + "── Command History ─────────────────────────────────" + RESET);
        System.out.printf("  %s%-4s  %-6s  %-20s  %-12s  %s%s%n",
            DIM, "#", "RISK", "STATUS", "TIME", "COMMAND", RESET);
        System.out.println(DIM + "  " + "─".repeat(75) + RESET);
        for (int i = 0; i < sessionEntries.size(); i++) {
            LogEntry e   = sessionEntries.get(i);
            String color = riskColor(e.risk());
            String time  = e.timestamp().substring(11); // HH:mm:ss only
            System.out.printf("  %s%-4d%s  %s%-6s%s  %s%-20s%s  %s%-12s%s  %s%n",
                DIM, i + 1, RESET,
                color, e.risk(), RESET,
                color, e.status(), RESET,
                DIM, time, RESET,
                e.command().length() > 60 ? e.command().substring(0, 57) + "..." : e.command());
        }
        System.out.println();
    }

    private void printStatus() {
        System.out.println();
        System.out.printf("  %sUser%s       : %s%n",          BOLD, RESET, username);
        System.out.printf("  %sTier%s       : %d (%s)%n",     BOLD, RESET, userTier, tierName(userTier));
        System.out.printf("  %sSession%s    : %s%n",          BOLD, RESET, sessionId);
        System.out.printf("  %sLog file%s   : %s%n",          BOLD, RESET, logFile);
        System.out.printf("  %sML engine%s  : %s%n",          BOLD, RESET,
            ml.isReady() ? GREEN + "ready (fine-tuned DistilBERT)" + RESET : YELLOW + "starting..." + RESET);
        System.out.printf("  %sKnown cmds%s : %d loaded%n%n", BOLD, RESET, knownCommands.size());
    }

    private void openLogs() {
        System.out.println();
        System.out.printf("  %sLog directory%s : %s%n", BOLD, RESET, LOG_DIR);
        System.out.printf("  %sCurrent log%s   : %s%n%n", BOLD, RESET, logFile);
        try {
            File[] logs = Paths.get(LOG_DIR).toFile().listFiles(f -> f.getName().endsWith(".jsonl"));
            if (logs == null || logs.length == 0) { printInfo("No session logs yet."); return; }
            Arrays.sort(logs, Comparator.comparingLong(File::lastModified).reversed());
            System.out.println(DIM + "  Recent sessions:" + RESET);
            for (int i = 0; i < Math.min(logs.length, 5); i++) {
                System.out.printf("    %s%s%s%n", i == 0 ? CYAN : DIM, logs[i].getName(), RESET);
            }
            System.out.println();
        } catch (Exception e) {
            printError("Could not list logs: " + e.getMessage());
        }
    }

    // ── Command Resolution ───────────────────────────────────────────────────

    /**
     * Longest-prefix match: tries up to MAX_TOKENS tokens, longest first.
     * Handles commands like "git reset --hard HEAD~3" (5 tokens) correctly.
     */
    private CommandDef resolveKnownCommand(String input) {
        String[] tokens = input.trim().split("\\s+");
        for (int len = Math.min(tokens.length, MAX_TOKENS); len >= 1; len--) {
            String key = String.join(" ", Arrays.copyOf(tokens, len));
            CommandDef def = knownCommands.get(key);
            if (def != null) return def;
        }
        return null;
    }

    private int inferTier(String command) {
        String cmd = command.trim().toLowerCase();
        if (TIER2_PATTERNS.stream().anyMatch(p -> p.matcher(cmd).find())) return 2;
        if (TIER1_PATTERNS.stream().anyMatch(p -> p.matcher(cmd).find())) return 1;
        return 0;
    }

    private static final List<Pattern> TIER2_PATTERNS = List.of(
        Pattern.compile("rm\\s+-[^\\s]*r[^\\s]*f"),
        Pattern.compile("\\bdd\\b.*of=/dev/"),
        Pattern.compile("\\bmkfs\\b"),      Pattern.compile("\\bvisudo\\b"),
        Pattern.compile("\\bshutdown\\b"),  Pattern.compile("\\breboot\\b"),
        Pattern.compile("\\bhalt\\b"),      Pattern.compile("iptables\\s+-F"),
        Pattern.compile("git\\s+push\\s+--force"),   Pattern.compile("git\\s+filter-branch"),
        Pattern.compile("git\\s+reflog\\s+expire"),  Pattern.compile("git\\s+clean\\s+-[^\\s]*f"),
        Pattern.compile("sudo\\s+passwd",            Pattern.CASE_INSENSITIVE),
        Pattern.compile("curl\\s+https?://[^\\s]*\\s*\\|\\s*sh", Pattern.CASE_INSENSITIVE),
        Pattern.compile("wget\\s+[^\\s]*\\s*\\|\\s*sh",          Pattern.CASE_INSENSITIVE),
        Pattern.compile("terraform\\s+(destroy|apply\\s+-auto-approve)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("kubectl\\s+delete",         Pattern.CASE_INSENSITIVE),
        Pattern.compile("docker\\s+system\\s+prune", Pattern.CASE_INSENSITIVE),
        Pattern.compile("ansible\\s+all\\s+-m\\s+shell.*reboot", Pattern.CASE_INSENSITIVE),
        Pattern.compile("aws\\s+(ec2\\s+terminate|rds\\s+delete|s3\\s+rb|iam\\s+delete)", Pattern.CASE_INSENSITIVE)
    );

    private static final List<Pattern> TIER1_PATTERNS = List.of(
        Pattern.compile("\\bsudo\\b"),       Pattern.compile("\\bchmod\\b"),
        Pattern.compile("\\bchown\\b"),      Pattern.compile("\\bkill\\b"),
        Pattern.compile("\\bapt\\b"),        Pattern.compile("\\bsystemctl\\b"),
        Pattern.compile("\\bcrontab\\b"),    Pattern.compile("\\bscp\\b"),
        Pattern.compile("terraform\\s+(apply|plan|import|init)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("ansible-playbook",  Pattern.CASE_INSENSITIVE),
        Pattern.compile("docker\\s+(rm|rmi|stop|restart|build|push|run)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("kubectl\\s+(apply|scale|set|rollout|create|label)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("aws\\s+(ec2|s3|iam|eks|rds|lambda)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("gcloud\\s+(compute|run|functions|builds)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("git\\s+(push|merge|rebase|cherry-pick|reset|commit\\s+--amend)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("helm\\s+(install|upgrade|uninstall|rollback)", Pattern.CASE_INSENSITIVE),
        Pattern.compile("vault\\s+kv\\s+(put|delete)", Pattern.CASE_INSENSITIVE)
    );

    // ── CSV Loading ──────────────────────────────────────────────────────────

    public void loadCommandsFromCsv(String path) {
        Path p = Paths.get(path);
        if (!Files.exists(p)) { printError("CSV not found: " + path); return; }
        int loaded = 0, skipped = 0;
        try (BufferedReader r = Files.newBufferedReader(p, StandardCharsets.UTF_8)) {
            String line;
            boolean firstLine = true;
            while ((line = r.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                // Skip header
                if (firstLine) {
                    firstLine = false;
                    if (line.toLowerCase().startsWith("command")) continue;
                }
                String[] parts = parseCsvLine(line);
                if (parts.length < 3) { skipped++; continue; }

                String cmd  = parts[0].trim();
                String desc = parts[1].trim();
                int    lvl;
                try { lvl = Integer.parseInt(parts[2].trim()); }
                catch (NumberFormatException e) { skipped++; continue; }
                if (lvl < 0 || lvl > 2 || cmd.isEmpty()) { skipped++; continue; }

                // CSV is authoritative — put (not putIfAbsent) so CSV wins
                knownCommands.put(cmd, new CommandDef(cmd, desc, lvl));
                loaded++;
            }
        } catch (IOException e) {
            printError("Failed to read CSV: " + e.getMessage()); return;
        }
        printInfo(String.format("Loaded %d commands from %s  (%d skipped)", loaded, path, skipped));
    }

    private String[] parseCsvLine(String line) {
        // Proper CSV parser: handles quoted fields with commas and escaped quotes inside
        List<String> fields = new ArrayList<>();
        StringBuilder sb    = new StringBuilder();
        boolean inQuotes    = false;
        char    prev        = 0;
        for (int i = 0; i < line.length(); i++) {
            char c = line.charAt(i);
            if (c == '"' && prev != '\\') {
                inQuotes = !inQuotes;
            } else if (c == ',' && !inQuotes) {
                fields.add(sb.toString().trim());
                sb.setLength(0);
            } else {
                sb.append(c);
            }
            prev = c;
        }
        fields.add(sb.toString().trim());
        return fields.toArray(new String[0]);
    }

    // ── Shutdown ─────────────────────────────────────────────────────────────

    private void shutdown() {
        isShuttingDown = true;
        System.out.println(CYAN + "\nFinalizing session..." + RESET);
        printSummary();

        // Get AI summary text for the log and session index
        String aiText = "(empty session)";
        if (!sessionEntries.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            for (LogEntry e : sessionEntries)
                sb.append(e.status()).append(": ").append(e.command())
                  .append(" [").append(e.risk()).append("]; ");
            aiText = ml.summarize(sb.toString());
        }
        writeSessionMeta("SESSION_END", aiText);
        writeSessionIndex(aiText); // final index write — marks session as no longer live
        ml.stop();
        System.out.println(DIM + "Log: " + logFile + RESET);
        System.out.println(GREEN + "Goodbye." + RESET);
        System.exit(0);
    }

    private void onShutdown() {
        if (!isShuttingDown) {
            // Abrupt termination (Ctrl+C) — graceful shutdown() wasn't called.
            // Mark session as dead in the index without blocking on AI summary.
            isShuttingDown = true;
            writeSessionIndex(sessionEntries.isEmpty() ? null : "(session terminated abruptly)");
        }
        ml.stop();
    }

    // ── UI ───────────────────────────────────────────────────────────────────

    private String buildPrompt() {
        String c = switch (userTier) {
            case 2  -> RED;
            case 1  -> YELLOW;
            default -> GREEN;
        };
        return String.format("%s[cmd.ai | %s | tier%d]%s$ ", c, username, userTier, RESET);
    }

    private void printBanner() {
        System.out.println();
        System.out.println(CYAN + BOLD +
            "   ██████╗ ██████╗ ███╗   ███╗███╗   ███╗ █████╗ ███╗   ██╗██████╗     █████╗ ██╗\n" +
            "  ██╔════╝██╔═══██╗████╗ ████║████╗ ████║██╔══██╗████╗  ██║██╔══██╗   ██╔══██╗██║\n" +
            "  ██║     ██║   ██║██╔████╔██║██╔████╔██║███████║██╔██╗ ██║██║  ██║   ███████║██║\n" +
            "  ██║     ██║   ██║██║╚██╔╝██║██║╚██╔╝██║██╔══██║██║╚██╗██║██║  ██║   ██╔══██║██║\n" +
            "  ╚██████╗╚██████╔╝██║ ╚═╝ ██║██║ ╚═╝ ██║██║  ██║██║ ╚████║██████╔╝   ██║  ██║██║\n" +
            "   ╚═════╝ ╚═════╝ ╚═╝     ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝    ╚═╝  ╚═╝╚═╝"
            + RESET);
        System.out.println(DIM + "  Terminal observability + privilege enforcement  v" + VERSION + RESET);
        System.out.println();
        System.out.printf("  %sUser%s   : %-20s %sTier%s : %d (%s)%n",
            BOLD, RESET, username, BOLD, RESET, userTier, tierName(userTier));
        System.out.printf("  %sSession%s: %s%n", BOLD, RESET, sessionId);
        if (group != null) System.out.printf("  %sGroup%s  : %s%n", BOLD, RESET, group);
        System.out.printf("  %sLog%s    : %s%n", BOLD, RESET, logFile);
        System.out.println();
        System.out.println(DIM + "  :help  :status  :history  :summary  :logs  :reload <csv>  exit" + RESET);
        System.out.println();
    }

    private static void printHelp() {
        System.out.println(CYAN + BOLD + "\n── Command.ai Help ─────────────────────────────────" + RESET);
        System.out.println("  Type any shell command — risk is assessed before execution.");
        System.out.println();
        System.out.println(BOLD + "  Built-in controls:" + RESET);
        System.out.println("    :help              Show this message");
        System.out.println("    :status            Session info and ML engine status");
        System.out.println("    :history           All commands run this session with timestamps");
        System.out.println("    :summary           AI-generated session summary");
        System.out.println("    :logs              Show log file location and recent sessions");
        System.out.println("    :reload <csv>      Hot-reload commands from a CSV file");
        System.out.println("    exit / quit        End session (writes summary to log)");
        System.out.println();
        System.out.println(BOLD + "  Startup flags:" + RESET);
        System.out.println("    --commands <path>  Load commands CSV at startup");
        System.out.println("    --user <name>      Set username for the session");
        System.out.println("    --tier <0|1|2>     Set privilege tier");
        System.out.println("    --group <name>     Tag this session for the dashboard (optional)");
        System.out.println("    --dashboard        Browse past and live sessions instead of starting one");
        System.out.println();
        System.out.println(BOLD + "  Privilege tiers:" + RESET);
        System.out.println("    0 = Standard   read-only, safe commands");
        System.out.println("    1 = Power User normal ops with side effects");
        System.out.println("    2 = Root/Admin all commands including destructive");
        System.out.println();
        System.out.println(BOLD + "  Risk levels:" + RESET);
        System.out.println("    " + RED    + "HIGH  " + RESET + " — Command blocked and logged");
        System.out.println("    " + YELLOW + "MEDIUM" + RESET + " — Warning shown, confirm to execute");
        System.out.println("    " + GREEN  + "LOW   " + RESET + " — Execute immediately");
        System.out.println();
        System.out.println(BOLD + "  Log format:" + RESET);
        System.out.println("    Logs at ~/.command_ai/logs/ — each session gets its own file.");
        System.out.println("    Each file contains both JSONL (machine-readable) and");
        System.out.println("    human-readable comment lines (prefixed with #).");
        System.out.println();
    }

    private void printInfo(String m)  { System.out.println(CYAN   + "  ℹ  " + m + RESET); }
    private void printWarn(String m)  { System.out.println(YELLOW  + "  ⚠  " + m + RESET); }
    private void printError(String m) { System.out.println(RED     + "  ✖  " + m + RESET); }
    private void printDim(String m)   { System.out.println(DIM     + m + RESET); }

    private String riskColor(String risk) {
        return switch (risk) { case "HIGH" -> RED; case "MEDIUM" -> YELLOW; default -> GREEN; };
    }

    private static String tierName(int tier) {
        return switch (tier) { case 2 -> "Root/Admin"; case 1 -> "Power User"; default -> "Standard"; };
    }

    private static String tierToRisk(int tier) {
        return switch (tier) { case 2 -> "HIGH"; case 1 -> "MEDIUM"; default -> "LOW"; };
    }

    private int promptTier() {
        System.out.println(CYAN + BOLD + "\nSelect your privilege tier:" + RESET);
        System.out.println("  0 = Standard   (safe / read-only commands only)");
        System.out.println("  1 = Power User (normal ops — push, deploy, install)");
        System.out.println("  2 = Root/Admin (all commands including destructive)");
        System.out.print(CYAN + "Tier [0/1/2]: " + RESET);
        try {
            String ans = new BufferedReader(new InputStreamReader(System.in)).readLine();
            int t = Integer.parseInt(ans.trim());
            if (t >= 0 && t <= 2) return t;
        } catch (Exception ignored) {}
        System.out.println(YELLOW + "Defaulting to tier 0 (Standard)." + RESET);
        return 0;
    }

    // ── Records ──────────────────────────────────────────────────────────────

    record CommandDef(String command, String description, int privilegeLevel) {}

    record LogEntry(String sessionId, String user, int tier, String timestamp,
                    String command, String status, String risk, String description,
                    int exitCode, String elapsed) {

        /** Machine-readable JSONL line */
        String toJsonLine() {
            return String.format(
                "{\"type\":\"CMD\",\"session\":\"%s\",\"user\":\"%s\",\"tier\":%d," +
                "\"ts\":\"%s\",\"cmd\":\"%s\",\"status\":\"%s\",\"risk\":\"%s\"," +
                "\"desc\":\"%s\",\"exit\":%d,\"elapsed\":\"%s\"}",
                sessionId, esc(user), tier, timestamp,
                esc(command), status, risk, esc(description),
                exitCode, elapsed != null ? elapsed : "");
        }

        /** Human-readable comment line written alongside each JSONL entry */
        String toHumanLine() {
            String time    = timestamp.substring(11); // HH:mm:ss
            String riskPad = String.format("%-6s", risk);
            String statPad = String.format("%-20s", status);
            String elapsed = this.elapsed != null ? "  [" + this.elapsed + "]" : "";
            return String.format("# %s  %s  %s  %s%s",
                time, riskPad, statPad,
                command.length() > 60 ? command.substring(0, 57) + "..." : command,
                elapsed);
        }

        private static String esc(String s) {
            if (s == null) return "";
            return s.replace("\\","\\\\").replace("\"","\\\"")
                    .replace("\n","\\n").replace("\r","\\r");
        }
    }

    // ── ML Engine ────────────────────────────────────────────────────────────

    static class MlEngine {
        private Process        process;
        private BufferedWriter toMl;
        private BufferedReader fromMl;
        private final AtomicBoolean ready = new AtomicBoolean(false);

        void start() {
            try {
                Path   script = extractScript();
                String py     = findPython();
                if (py == null) {
                    System.err.println("[ML] Python3 not found — rule-based fallback active.");
                    return;
                }
                process = new ProcessBuilder(py, script.toString())
                    .redirectErrorStream(false).start();
                toMl   = new BufferedWriter(new OutputStreamWriter(
                    process.getOutputStream(), StandardCharsets.UTF_8));
                fromMl = new BufferedReader(new InputStreamReader(
                    process.getInputStream(), StandardCharsets.UTF_8));

                System.out.print("[ML] Loading fine-tuned DistilBERT... ");
                CompletableFuture.runAsync(() -> {
                    try {
                        String line = fromMl.readLine();
                        if (line != null && line.contains("ready")) ready.set(true);
                    } catch (IOException ignored) {}
                }).get(90, TimeUnit.SECONDS);
                System.out.println(ready.get() ? "ready." : "timeout — rule-based fallback active.");
            } catch (Exception e) {
                System.err.println("[ML] " + e.getMessage() + " — rule-based fallback active.");
            }
        }

        /** Send command to ML for risk classification */
        String assess(String command) {
            if (!ready.get()) return ruleBasedRisk(command);
            try {
                // Always check rules first — they're faster and handle known-bad patterns
                String ruleResult = ruleBasedOverride(command);
                if (ruleResult != null) return ruleResult;

                String json = "{\"action\":\"assess\",\"data\":" + jsonStr(command) + "}";
                toMl.write(json); toMl.newLine(); toMl.flush();
                String resp = fromMl.readLine();
                if (resp == null) return ruleBasedRisk(command);
                return extractStr(resp, "result", ruleBasedRisk(command));
            } catch (IOException e) {
                return ruleBasedRisk(command);
            }
        }

        /** Send session to ML for summarization */
        String summarize(String sessionText) {
            if (!ready.get()) return "(ML not available)";
            try {
                String json = "{\"action\":\"summarize\",\"data\":" + jsonStr(sessionText) + "}";
                toMl.write(json); toMl.newLine(); toMl.flush();
                String resp = fromMl.readLine();
                if (resp == null || resp.isEmpty()) return "(summary unavailable)";

                // Extract "result" field manually — handles unicode escapes correctly
                int start = resp.indexOf("\"result\": \"");
                if (start < 0) start = resp.indexOf("\"result\":\"");
                if (start < 0) return "(could not parse summary)";

                start = resp.indexOf("\"", start + 9) + 1;
                StringBuilder sb  = new StringBuilder();
                boolean escaped   = false;
                for (int i = start; i < resp.length(); i++) {
                    char c = resp.charAt(i);
                    if (escaped) {
                        switch (c) {
                            case '"'  -> sb.append('"');
                            case 'n'  -> sb.append(' ');
                            case 'u'  -> {
                                // unicode escape sequence (4 hex digits follow)
                                if (i + 4 < resp.length()) {
                                    String hex = resp.substring(i + 1, i + 5);
                                    try {
                                        sb.append((char) Integer.parseInt(hex, 16));
                                        i += 4;
                                    } catch (NumberFormatException ex) {
                                        sb.append("\\u").append(hex);
                                    }
                                }
                            }
                            default   -> sb.append('\\').append(c);
                        }
                        escaped = false;
                    } else if (c == '\\') {
                        escaped = true;
                    } else if (c == '"') {
                        break; // end of string
                    } else {
                        sb.append(c);
                    }
                }
                return sb.toString().isEmpty() ? "(summary unavailable)" : sb.toString();
            } catch (IOException e) {
                return "(summary error: " + e.getMessage() + ")";
            }
        }

        boolean isReady() { return ready.get(); }

        void stop() {
            try {
                if (toMl != null) {
                    toMl.write("{\"action\":\"quit\"}"); toMl.newLine(); toMl.flush();
                }
            } catch (IOException ignored) {}
            try { if (process != null) process.destroyForcibly(); } catch (Exception ignored) {}
        }

        // ── Rule-based patterns ───────────────────────────────────────────────

        /** Returns HIGH if command matches known-dangerous patterns, null otherwise. */
        public static String ruleBasedOverride(String cmd) {
            String c = cmd.toLowerCase();
            for (Pattern p : HIGH_P) if (p.matcher(c).find()) return "HIGH";
            return null; // let model decide for everything else
        }

        static String ruleBasedRisk(String cmd) {
            String c = cmd.toLowerCase();
            for (Pattern p : HIGH_P)   if (p.matcher(c).find()) return "HIGH";
            for (Pattern p : MEDIUM_P) if (p.matcher(c).find()) return "MEDIUM";
            return "LOW";
        }

        private static final List<Pattern> HIGH_P = List.of(
            Pattern.compile("rm\\s+-[^\\s]*r[^\\s]*f",         Pattern.CASE_INSENSITIVE),
            Pattern.compile("\\bdd\\b.*of=/dev/"),
            Pattern.compile("\\bmkfs\\b"),       Pattern.compile("\\bvisudo\\b"),
            Pattern.compile("\\bshutdown\\b"),   Pattern.compile("\\breboot\\b"),
            Pattern.compile("\\bhalt\\b"),       Pattern.compile("iptables\\s+-F"),
            Pattern.compile("git\\s+push\\s+--force"),      Pattern.compile("git\\s+filter-branch"),
            Pattern.compile("git\\s+reflog\\s+expire"),     Pattern.compile("git\\s+clean\\s+-[^\\s]*f"),
            Pattern.compile("git\\s+reset\\s+--hard\\s+head[~^]", Pattern.CASE_INSENSITIVE),
            Pattern.compile("sudo\\s+passwd",                Pattern.CASE_INSENSITIVE),
            Pattern.compile("curl\\s+https?://[^\\s]*\\|\\s*sh", Pattern.CASE_INSENSITIVE),
            Pattern.compile("wget\\s+[^\\s]*\\|\\s*sh",     Pattern.CASE_INSENSITIVE),
            Pattern.compile(":\\(\\)\\s*\\{.*:\\s*\\|.*:\\s*\\}"),
            Pattern.compile("terraform\\s+destroy",          Pattern.CASE_INSENSITIVE),
            Pattern.compile("kubectl\\s+delete",             Pattern.CASE_INSENSITIVE),
            Pattern.compile("docker\\s+system\\s+prune",     Pattern.CASE_INSENSITIVE),
            Pattern.compile("aws\\s+ec2\\s+terminate",       Pattern.CASE_INSENSITIVE),
            Pattern.compile("aws\\s+rds\\s+delete",          Pattern.CASE_INSENSITIVE),
            Pattern.compile("aws\\s+s3\\s+rb",               Pattern.CASE_INSENSITIVE),
            Pattern.compile("redis-cli\\s+(flushall|flushdb|shutdown)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("mongo.*dropdatabase",           Pattern.CASE_INSENSITIVE),
            Pattern.compile("psql.*drop\\s+(database|table|role)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("mysql.*drop\\s+(database|table|user)", Pattern.CASE_INSENSITIVE)
        );

        private static final List<Pattern> MEDIUM_P = List.of(
            Pattern.compile("\\bsudo\\b"),       Pattern.compile("\\bchmod\\b"),
            Pattern.compile("\\bchown\\b"),      Pattern.compile("\\bkill\\b"),
            Pattern.compile("\\bapt\\b"),        Pattern.compile("\\bsystemctl\\b"),
            Pattern.compile("\\bcrontab\\b"),    Pattern.compile("\\bscp\\b"),
            Pattern.compile("git\\s+(push|merge|rebase|cherry-pick)"),
            Pattern.compile("git\\s+reset\\s+--hard"),
            Pattern.compile("git\\s+commit\\s+--amend"),
            Pattern.compile("terraform\\s+(apply|import)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("ansible-playbook",             Pattern.CASE_INSENSITIVE),
            Pattern.compile("docker\\s+(rm|rmi|stop|build|push)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("kubectl\\s+(apply|scale|set)",  Pattern.CASE_INSENSITIVE),
            Pattern.compile("helm\\s+(install|upgrade|uninstall)", Pattern.CASE_INSENSITIVE),
            Pattern.compile("curl.*\\|\\s*sh"),  Pattern.compile("wget.*\\|\\s*sh")
        );

        // ── Helpers ───────────────────────────────────────────────────────────

        /** Serialize a string as a JSON string value, handling all special chars */
        private static String jsonStr(String s) {
            if (s == null) return "\"\"";
            StringBuilder sb = new StringBuilder("\"");
            for (char c : s.toCharArray()) {
                switch (c) {
                    case '"'  -> sb.append("\\\"");
                    case '\\' -> sb.append("\\\\");
                    case '\n' -> sb.append("\\n");
                    case '\r' -> sb.append("\\r");
                    case '\t' -> sb.append("\\t");
                    default   -> {
                        if (c < 0x20) sb.append(String.format("\\u%04x", (int) c));
                        else sb.append(c);
                    }
                }
            }
            sb.append("\"");
            return sb.toString();
        }

        private static String extractStr(String json, String key, String fallback) {
            // Handles both "key":"val" and "key": "val"
            for (String search : List.of("\"" + key + "\": \"", "\"" + key + "\":\"")) {
                int s = json.indexOf(search);
                if (s >= 0) {
                    s += search.length();
                    int e = json.indexOf("\"", s);
                    return e > s ? json.substring(s, e) : fallback;
                }
            }
            return fallback;
        }

        private Path extractScript() throws IOException {
            Path dir    = Paths.get(System.getProperty("user.home"), ".command_ai");
            Files.createDirectories(dir);
            Path target = dir.resolve(ML_SCRIPT);
            try (InputStream is = CommandAI.class.getResourceAsStream("/" + ML_SCRIPT)) {
                if (is == null) throw new IOException("ml_engine.py not found in JAR");
                Files.copy(is, target, StandardCopyOption.REPLACE_EXISTING);
            }
            return target;
        }

        private String findPython() {
            // On macOS with venv, also try the venv python directly
            List<String> candidates = List.of(
                System.getProperty("user.home") + "/.command_ai/venv/bin/python3",
                "python3", "python"
            );
            for (String c : candidates) {
                try {
                    Process p = new ProcessBuilder(c, "--version").start();
                    if (p.waitFor(3, TimeUnit.SECONDS) && p.exitValue() == 0) return c;
                } catch (Exception ignored) {}
            }
            return null;
        }
    }


    // ── Dashboard ────────────────────────────────────────────────────────────

    /**
     * Local web dashboard for browsing live and past sessions.
     * Starts a tiny HTTP server (JDK built-in, no extra dependency) on
     * localhost, serves a single-page HTML UI, and opens it in the
     * default browser. Reads ~/.command_ai/sessions/*.json fresh on every
     * request, so the in-page Refresh button always reflects current state.
     *
     * Sessions are independent by default — --group is purely optional
     * metadata for the developer's own organization, never required.
     */
    static class Dashboard {

        private static final int   PORT_START = 7878;
        private static final int   PORT_RANGE = 20; // try 7878-7897 if busy

        record SessionInfo(
            String session, String user, int tier, String group,
            long pid, boolean alive, String started, String updated,
            int totalCommands, int blocked, int highRisk,
            String logFile, String summary
        ) {}

        static void run() throws IOException {
            com.sun.net.httpserver.HttpServer server = startServer();
            int port = server.getAddress().getPort();
            String url = "http://localhost:" + port + "/";

            System.out.println();
            System.out.println("\033[0;36m\033[1mCommand.ai dashboard running at " + url + "\033[0m");
            System.out.println("\033[2mPress Ctrl+C to stop.\033[0m");
            System.out.println();

            openBrowser(url);

            // Block forever — server runs on its own thread pool until Ctrl+C
            try { Thread.currentThread().join(); } catch (InterruptedException ignored) {}
        }

        private static com.sun.net.httpserver.HttpServer startServer() throws IOException {
            IOException lastError = null;
            for (int i = 0; i < PORT_RANGE; i++) {
                int port = PORT_START + i;
                try {
                    var server = com.sun.net.httpserver.HttpServer.create(
                        new java.net.InetSocketAddress("localhost", port), 0);
                    server.createContext("/",         Dashboard::handleIndex);
                    server.createContext("/api/sessions", Dashboard::handleSessionsApi);
                    server.createContext("/api/log",      Dashboard::handleLogApi);
                    server.createContext("/api/group",    Dashboard::handleGroupApi);
                    server.createContext("/api/group/rename", Dashboard::handleGroupRenameApi);
                    server.createContext("/api/kill",     Dashboard::handleKillApi);
                    server.createContext("/api/launch",   Dashboard::handleLaunchApi);
                    server.setExecutor(java.util.concurrent.Executors.newFixedThreadPool(4));
                    server.start();
                    return server;
                } catch (IOException e) {
                    lastError = e; // port busy, try next
                }
            }
            throw lastError != null ? lastError : new IOException("Could not start dashboard server");
        }

        private static void openBrowser(String url) {
            try {
                String os = System.getProperty("os.name").toLowerCase();
                ProcessBuilder pb = os.contains("mac")
                    ? new ProcessBuilder("open", url)
                    : os.contains("win")
                        ? new ProcessBuilder("cmd", "/c", "start", url)
                        : new ProcessBuilder("xdg-open", url);
                pb.start();
            } catch (Exception e) {
                System.out.println("\033[0;33mCould not auto-open browser — open " + url + " manually.\033[0m");
            }
        }

        // ── HTTP Handlers ────────────────────────────────────────────────────

        private static void handleIndex(com.sun.net.httpserver.HttpExchange ex) throws IOException {
            byte[] body = loadDashboardHtml().getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().add("Content-Type", "text/html; charset=utf-8");
            ex.sendResponseHeaders(200, body.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(body); }
        }

        private static void handleSessionsApi(com.sun.net.httpserver.HttpExchange ex) throws IOException {
            List<SessionInfo> sessions = loadSessions();
            StringBuilder json = new StringBuilder("[");
            for (int i = 0; i < sessions.size(); i++) {
                if (i > 0) json.append(",");
                json.append(toJson(sessions.get(i)));
            }
            json.append("]");

            byte[] body = json.toString().getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().add("Content-Type", "application/json");
            ex.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
            ex.sendResponseHeaders(200, body.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(body); }
        }

        private static void handleLogApi(com.sun.net.httpserver.HttpExchange ex) throws IOException {
            String query = ex.getRequestURI().getQuery();
            String sessionId = null;
            if (query != null) {
                for (String pair : query.split("&")) {
                    String[] kv = pair.split("=", 2);
                    if (kv.length == 2 && kv[0].equals("session")) {
                        sessionId = java.net.URLDecoder.decode(kv[1], StandardCharsets.UTF_8);
                    }
                }
            }

            String content;
            if (sessionId == null || !sessionId.matches("[A-Za-z0-9_]+")) {
                content = "Invalid session id.";
            } else {
                Path logPath = Paths.get(LOG_DIR, "session_" + sessionId + ".jsonl");
                content = Files.exists(logPath)
                    ? Files.readString(logPath, StandardCharsets.UTF_8)
                    : "Log file not found: " + logPath;
            }

            byte[] body = content.getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().add("Content-Type", "text/plain; charset=utf-8");
            ex.sendResponseHeaders(200, body.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(body); }
        }

        /**
         * POST /api/group  body: {"session":"<id>","group":"<name>"|null}
         * Rewrites a single session's index file with a new group value.
         * Pass group as empty string or null to ungroup.
         * Persists even if the originating terminal has already exited.
         */
        private static void handleGroupApi(com.sun.net.httpserver.HttpExchange ex) throws IOException {
            if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                ex.sendResponseHeaders(405, -1);
                return;
            }

            String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            String sessionId = field(body, "session");
            String newGroup  = field(body, "group");

            if (sessionId == null || !sessionId.matches("[A-Za-z0-9_]+")) {
                sendJson(ex, 400, "{\"error\":\"invalid session id\"}");
                return;
            }

            SessionInfo updated = writeSessionGroup(sessionId, newGroup);
            if (updated == null) {
                sendJson(ex, 404, "{\"error\":\"session not found\"}");
                return;
            }
            sendJson(ex, 200, toJson(updated));
        }

        /**
         * POST /api/group/rename  body: {"from":"<old group name>","to":"<new group name>"}
         * Renames a group across every session currently in it. Pass "to" as
         * empty string to ungroup every session in "from" (i.e. delete the group).
         */
        private static void handleGroupRenameApi(com.sun.net.httpserver.HttpExchange ex) throws IOException {
            if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                ex.sendResponseHeaders(405, -1);
                return;
            }

            String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            String from = field(body, "from");
            String to   = field(body, "to");

            if (from == null || from.isBlank()) {
                sendJson(ex, 400, "{\"error\":\"missing 'from' group name\"}");
                return;
            }

            List<SessionInfo> all = loadSessions();
            int updatedCount = 0;
            for (SessionInfo s : all) {
                if (from.equals(s.group())) {
                    if (writeSessionGroup(s.session(), to) != null) updatedCount++;
                }
            }

            sendJson(ex, 200, String.format("{\"updated\":%d}", updatedCount));
        }

        /**
         * POST /api/kill  body: {"session":"<id>"}
         * Gracefully terminates a live session:
         *   1. Sends SIGTERM to the commandai Java process — triggers its
         *      shutdown hook which writes final log + session index.
         *   2. Waits up to 3 seconds for it to exit cleanly.
         *   3. If still alive, sends SIGKILL (force kill).
         *   4. Closes the terminal window on macOS via AppleScript by
         *      finding the window whose shell has our Java process as a
         *      descendant. On Linux, kills the parent process (terminal emulator).
         *   5. Marks the session dead in the index so the dashboard updates.
         */
        private static void handleKillApi(com.sun.net.httpserver.HttpExchange ex) throws IOException {
            if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                ex.sendResponseHeaders(405, -1);
                return;
            }
            String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            String sessionId = field(body, "session");
            if (sessionId == null || !sessionId.matches("[A-Za-z0-9_]+")) {
                sendJson(ex, 400, "{\"error\":\"invalid session id\"}");
                return;
            }

            Path indexFile = Paths.get(SESSION_DIR, sessionId + ".json");
            if (!Files.exists(indexFile)) {
                sendJson(ex, 404, "{\"error\":\"session not found\"}");
                return;
            }
            SessionInfo info = parseSessionJson(Files.readString(indexFile, StandardCharsets.UTF_8));
            if (info == null || info.pid() <= 0) {
                sendJson(ex, 500, "{\"error\":\"could not resolve pid\"}");
                return;
            }

            long javaPid = info.pid();
            boolean killed = killProcessGracefully(javaPid);
            closeTerminalWindow(javaPid);

            // Mark dead in index so dashboard reflects the change immediately,
            // regardless of whether the shutdown hook finished writing its own copy.
            SessionInfo deadInfo = new SessionInfo(
                info.session(), info.user(), info.tier(), info.group(),
                info.pid(), false, info.started(), info.updated(),
                info.totalCommands(), info.blocked(), info.highRisk(),
                info.logFile(), info.summary() != null
                    ? info.summary()
                    : "(session stopped from dashboard)"
            );
            writeSessionFile(indexFile, deadInfo);

            sendJson(ex, 200, String.format("{\"killed\":%b}", killed));
        }

        /**
         * Sends SIGTERM, waits up to 3 seconds for clean exit, then SIGKILL.
         * Returns true if the process is no longer alive after this.
         */
        private static boolean killProcessGracefully(long pid) {
            Optional<ProcessHandle> handleOpt = ProcessHandle.of(pid);
            if (handleOpt.isEmpty() || !handleOpt.get().isAlive()) return true; // already gone

            ProcessHandle handle = handleOpt.get();
            handle.destroy(); // SIGTERM — lets shutdown hook write final log

            // Wait up to 3 seconds for graceful exit
            long deadline = System.currentTimeMillis() + 3000;
            while (handle.isAlive() && System.currentTimeMillis() < deadline) {
                try { Thread.sleep(200); } catch (InterruptedException ignored) {}
            }

            // Force kill if still running
            if (handle.isAlive()) handle.destroyForcibly();
            return !handle.isAlive();
        }

        /**
         * Attempts to close the terminal window that contains the given PID.
         *
         * macOS: uses AppleScript to find and close the Terminal.app tab
         *        whose shell process has our Java PID as a descendant.
         *        Walks up the process tree from the Java PID to find the
         *        shell (bash/zsh) that Terminal.app tracks per-tab.
         *
         * Linux: finds the terminal emulator that is an ancestor of our
         *        Java process and sends SIGHUP to it, which closes the window.
         */
        private static void closeTerminalWindow(long javaPid) {
            String os = System.getProperty("os.name", "").toLowerCase();
            try {
                if (os.contains("mac")) {
                    closeTerminalWindowMac(javaPid);
                } else if (!os.contains("win")) {
                    closeTerminalWindowLinux(javaPid);
                }
                // Windows: the cmd /c start window will close on its own
                // when the Java process exits via killProcessGracefully above
            } catch (Exception ignored) {
                // Best-effort — non-fatal if we can't close the window
            }
        }

        private static void closeTerminalWindowMac(long javaPid) throws IOException, InterruptedException {
            // Walk up the process tree from our Java process to find the
            // shell (bash or zsh) that Terminal.app spawned for this tab.
            long shellPid = findAncestorShellPid(javaPid);

            // AppleScript: iterate Terminal windows/tabs, find the one whose
            // tty's shell PID matches, then close that tab (or window if it's the only tab).
            String script = String.format("""
                tell application "Terminal"
                    repeat with w in windows
                        repeat with t in tabs of w
                            if tty of t is not "" then
                                try
                                    set tabPid to do shell script "ps -o pid= -t " & tty of t & " 2>/dev/null | head -1 | tr -d ' '"
                                    if tabPid = "%d" or tabPid = "%d" then
                                        if (count of tabs of w) > 1 then
                                            close t
                                        else
                                            close w
                                        end if
                                        return
                                    end if
                                end try
                            end if
                        end repeat
                    end repeat
                end tell
                """, shellPid, javaPid);

            Process p = new ProcessBuilder("osascript", "-e", script).start();
            p.waitFor(5, java.util.concurrent.TimeUnit.SECONDS);
        }

        /** Walks the process tree from a given PID upward to find a bash/zsh ancestor. */
        private static long findAncestorShellPid(long pid) {
            long current = pid;
            for (int depth = 0; depth < 6; depth++) {
                Optional<ProcessHandle> h = ProcessHandle.of(current);
                if (h.isEmpty()) break;
                Optional<String> cmd = h.get().info().command();
                if (cmd.isPresent()) {
                    String name = Paths.get(cmd.get()).getFileName().toString().toLowerCase();
                    if (name.equals("bash") || name.equals("zsh") || name.equals("sh")) {
                        return current;
                    }
                }
                Optional<ProcessHandle> parent = h.get().parent();
                if (parent.isEmpty()) break;
                current = parent.get().pid();
            }
            return pid; // fallback: use the Java PID itself
        }

        private static void closeTerminalWindowLinux(long javaPid) {
            // Walk up to find a known terminal emulator process and send SIGHUP
            // which causes it to close the window (same as user closing window).
            Set<String> terminalEmulators = Set.of(
                "gnome-terminal", "gnome-terminal-server", "konsole",
                "xterm", "xfce4-terminal", "lxterminal", "tilix",
                "alacritty", "kitty", "wezterm"
            );

            long current = javaPid;
            for (int depth = 0; depth < 8; depth++) {
                Optional<ProcessHandle> h = ProcessHandle.of(current);
                if (h.isEmpty()) break;
                Optional<String> cmd = h.get().info().command();
                if (cmd.isPresent()) {
                    String name = Paths.get(cmd.get()).getFileName().toString().toLowerCase();
                    if (terminalEmulators.contains(name)) {
                        try {
                            // SIGHUP closes the terminal window cleanly
                            new ProcessBuilder("kill", "-HUP", String.valueOf(current)).start();
                        } catch (IOException ignored) {}
                        return;
                    }
                }
                Optional<ProcessHandle> parent = h.get().parent();
                if (parent.isEmpty()) break;
                current = parent.get().pid();
            }
        }

        /**
         * POST /api/launch  body: {"user":"<name>","tier":0|1|2,"group":"<name>"|null,"commands":"<csv path>"|null}
         * Opens a brand-new OS terminal window running this same JAR with the
         * given flags, so a fresh tracked session starts. Platform-specific:
         * macOS uses Terminal.app via AppleScript, Linux tries common terminal
         * emulators, Windows uses cmd /c start.
         */
        private static void handleLaunchApi(com.sun.net.httpserver.HttpExchange ex) throws IOException {
            if (!"POST".equalsIgnoreCase(ex.getRequestMethod())) {
                ex.sendResponseHeaders(405, -1);
                return;
            }
            String body = new String(ex.getRequestBody().readAllBytes(), StandardCharsets.UTF_8);
            String user     = field(body, "user");
            String group    = field(body, "group");
            String commands = field(body, "commands");
            int    tier     = intField(body, "tier", 0);

            String jarPath = resolveJarPath();
            if (jarPath == null) {
                sendJson(ex, 500, "{\"error\":\"could not locate running JAR — launch only works when running from the packaged commandai.jar\"}");
                return;
            }

            StringBuilder cmd = new StringBuilder("java -jar ").append(quote(jarPath));
            if (user != null && !user.isBlank())     cmd.append(" --user ").append(quote(user));
            cmd.append(" --tier ").append(Math.max(0, Math.min(2, tier)));
            if (group != null && !group.isBlank())    cmd.append(" --group ").append(quote(group));
            if (commands != null && !commands.isBlank()) cmd.append(" --commands ").append(quote(commands));

            boolean opened = openTerminalWith(cmd.toString());
            if (opened) sendJson(ex, 200, "{\"launched\":true}");
            else        sendJson(ex, 500, "{\"error\":\"could not open a terminal window on this OS\"}");
        }

        /** Best-effort resolution of the currently-running JAR's filesystem path. */
        private static String resolveJarPath() {
            try {
                Path p = Paths.get(Dashboard.class.getProtectionDomain()
                    .getCodeSource().getLocation().toURI());
                if (Files.exists(p) && p.toString().endsWith(".jar")) return p.toAbsolutePath().toString();
            } catch (Exception ignored) {}
            return null;
        }

        private static String quote(String s) {
            return "'" + s.replace("'", "'\\''") + "'";
        }

        /** Opens a new terminal window and runs the given shell command in it. */
        private static boolean openTerminalWith(String shellCommand) {
            String os = System.getProperty("os.name").toLowerCase();
            try {
                if (os.contains("mac")) {
                    String script = "tell application \"Terminal\" to do script " +
                        appleScriptQuote(shellCommand);
                    new ProcessBuilder("osascript", "-e", script).start();
                    return true;
                } else if (os.contains("win")) {
                    new ProcessBuilder("cmd", "/c", "start", "cmd", "/k", shellCommand).start();
                    return true;
                } else {
                    // Linux — try common terminal emulators in order until one launches
                    for (String[] candidate : new String[][] {
                        {"gnome-terminal", "--", "bash", "-c", shellCommand + "; exec bash"},
                        {"konsole", "-e", "bash", "-c", shellCommand + "; exec bash"},
                        {"xterm", "-e", "bash", "-c", shellCommand + "; exec bash"},
                        {"x-terminal-emulator", "-e", "bash", "-c", shellCommand + "; exec bash"}
                    }) {
                        try {
                            new ProcessBuilder(candidate).start();
                            return true;
                        } catch (IOException ignored) { /* try next */ }
                    }
                    return false;
                }
            } catch (IOException e) {
                return false;
            }
        }

        private static String appleScriptQuote(String s) {
            return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
        }

        /** Shared rewrite logic for a single session's group field. Returns null if session not found. */
        private static SessionInfo writeSessionGroup(String sessionId, String newGroup) throws IOException {
            Path indexFile = Paths.get(SESSION_DIR, sessionId + ".json");
            if (!Files.exists(indexFile)) return null;

            String existing = Files.readString(indexFile, StandardCharsets.UTF_8);
            SessionInfo old  = parseSessionJson(existing);
            if (old == null) return null;

            String groupToSet = (newGroup == null || newGroup.isBlank()) ? null : newGroup.trim();
            SessionInfo updated = new SessionInfo(
                old.session(), old.user(), old.tier(), groupToSet,
                old.pid(), old.alive(), old.started(), old.updated(),
                old.totalCommands(), old.blocked(), old.highRisk(),
                old.logFile(), old.summary()
            );

            writeSessionFile(indexFile, updated);
            return updated;
        }

        /** Serializes a SessionInfo back to its index JSON file on disk. */
        private static void writeSessionFile(Path indexFile, SessionInfo s) throws IOException {
            String json = String.format(
                "{\"session\":\"%s\",\"user\":\"%s\",\"tier\":%d,\"group\":%s," +
                "\"pid\":%d,\"alive\":%b,\"started\":\"%s\",\"updated\":\"%s\"," +
                "\"total_commands\":%d,\"blocked\":%d,\"high_risk\":%d," +
                "\"log_file\":\"%s\",\"summary\":%s}",
                jsonEsc(s.session()), jsonEsc(s.user()), s.tier(),
                s.group() != null ? "\"" + jsonEsc(s.group()) + "\"" : "null",
                s.pid(), s.alive(), jsonEsc(s.started()), jsonEsc(s.updated()),
                s.totalCommands(), s.blocked(), s.highRisk(),
                jsonEsc(s.logFile()),
                s.summary() != null ? "\"" + jsonEsc(s.summary()) + "\"" : "null"
            );
            Files.writeString(indexFile, json, StandardCharsets.UTF_8,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        }

        private static void sendJson(com.sun.net.httpserver.HttpExchange ex, int status, String json) throws IOException {
            byte[] body = json.getBytes(StandardCharsets.UTF_8);
            ex.getResponseHeaders().add("Content-Type", "application/json");
            ex.sendResponseHeaders(status, body.length);
            try (OutputStream os = ex.getResponseBody()) { os.write(body); }
        }

        // ── Session loading (shared with terminal mode previously) ────────────

        private static List<SessionInfo> loadSessions() {
            Path dir = Paths.get(SESSION_DIR);
            List<SessionInfo> result = new ArrayList<>();
            if (!Files.exists(dir)) return result;

            try (var stream = Files.list(dir)) {
                for (Path p : stream.filter(f -> f.toString().endsWith(".json")).toList()) {
                    try {
                        String json = Files.readString(p, StandardCharsets.UTF_8);
                        SessionInfo info = parseSessionJson(json);
                        if (info != null) result.add(info);
                    } catch (IOException ignored) {}
                }
            } catch (IOException ignored) {}
            return result;
        }

        private static String toJson(SessionInfo s) {
            return String.format(
                "{\"session\":\"%s\",\"user\":\"%s\",\"tier\":%d,\"group\":%s," +
                "\"pid\":%d,\"alive\":%b,\"started\":\"%s\",\"updated\":\"%s\"," +
                "\"totalCommands\":%d,\"blocked\":%d,\"highRisk\":%d,\"summary\":%s}",
                jsonEsc(s.session()), jsonEsc(s.user()), s.tier(),
                s.group() != null ? "\"" + jsonEsc(s.group()) + "\"" : "null",
                s.pid(), s.alive(), jsonEsc(s.started()), jsonEsc(s.updated()),
                s.totalCommands(), s.blocked(), s.highRisk(),
                s.summary() != null ? "\"" + jsonEsc(s.summary()) + "\"" : "null"
            );
        }

        private static String jsonEsc(String s) {
            if (s == null) return "";
            return s.replace("\\","\\\\").replace("\"","\\\"")
                    .replace("\n","\\n").replace("\r","\\r");
        }

        private static SessionInfo parseSessionJson(String json) {
            String session = field(json, "session");
            if (session == null) return null;
            return new SessionInfo(
                session, field(json, "user"), intField(json, "tier", 0),
                field(json, "group"), (long) intField(json, "pid", 0),
                boolField(json, "alive"), field(json, "started"), field(json, "updated"),
                intField(json, "total_commands", 0), intField(json, "blocked", 0),
                intField(json, "high_risk", 0), field(json, "log_file"), field(json, "summary")
            );
        }

        private static String field(String json, String key) {
            for (String search : List.of("\"" + key + "\":\"", "\"" + key + "\": \"")) {
                int s = json.indexOf(search);
                if (s >= 0) {
                    s += search.length();
                    StringBuilder sb = new StringBuilder();
                    boolean escaped = false;
                    for (int i = s; i < json.length(); i++) {
                        char c = json.charAt(i);
                        if (escaped) { sb.append(c); escaped = false; }
                        else if (c == '\\') escaped = true;
                        else if (c == '"') break;
                        else sb.append(c);
                    }
                    return sb.toString();
                }
            }
            return null;
        }

        private static int intField(String json, String key, int fallback) {
            for (String search : List.of("\"" + key + "\":", "\"" + key + "\": ")) {
                int s = json.indexOf(search);
                if (s >= 0) {
                    s += search.length();
                    int e = s;
                    while (e < json.length() && (Character.isDigit(json.charAt(e)) || json.charAt(e) == '-')) e++;
                    if (e > s) {
                        try { return Integer.parseInt(json.substring(s, e)); }
                        catch (NumberFormatException ignored) {}
                    }
                }
            }
            return fallback;
        }

        private static boolean boolField(String json, String key) {
            for (String search : List.of("\"" + key + "\":", "\"" + key + "\": ")) {
                int s = json.indexOf(search);
                if (s >= 0) return json.startsWith("true", s + search.length());
            }
            return false;
        }

        // ── HTML — loaded from bundled resource, not inlined in Java ──────────

        private static final String DASHBOARD_RESOURCE = "dashboard.html";

        private static String loadDashboardHtml() throws IOException {
            try (InputStream is = CommandAI.class.getResourceAsStream("/" + DASHBOARD_RESOURCE)) {
                if (is == null) throw new IOException(DASHBOARD_RESOURCE + " not found in JAR resources");
                return new String(is.readAllBytes(), StandardCharsets.UTF_8);
            }
        }
    }
}