#!/usr/bin/env bash
set -e

ROOT_DIR="$HOME/codepilot_free"
EXT_DIR="$ROOT_DIR/vscode-codepilot"
SERVER_DIR="$ROOT_DIR/server"
LOG_FILE="$SERVER_DIR/server.log"
LAUNCHER="/usr/local/bin/codepilot"

echo ">>> Installing CodePilot into $ROOT_DIR ..."

mkdir -p "$ROOT_DIR" "$EXT_DIR" "$SERVER_DIR"

#######################################
# VS Code Extension
#######################################
echo ">>> Setting up VS Code extension..."
cd "$EXT_DIR"
npm init -y >/dev/null
npm install --save node-fetch@2 >/dev/null
npm install --save-dev typescript vsce @types/node >/dev/null

# package.json
cat > package.json <<EOF
{
  "name": "codepilot-free",
  "displayName": "CodePilot Free",
  "version": "0.3.0",
  "publisher": "local",
  "engines": { "vscode": "^1.80.0" },
  "categories": ["Programming Languages"],
  "main": "out/extension.js",
  "contributes": {
    "commands": [{ "command": "codepilot.start", "title": "Start CodePilot" }]
  },
  "activationEvents": ["onStartupFinished"]
}
EOF

# tsconfig.json
cat > tsconfig.json <<EOF
{
  "compilerOptions": {
    "module": "commonjs",
    "target": "es6",
    "outDir": "out",
    "lib": ["es6"],
    "strict": true,
    "esModuleInterop": true
  }
}
EOF

# extension.ts
mkdir -p src
cat > src/extension.ts <<'EOF'
import * as vscode from "vscode";
import fetch from "node-fetch";

export function activate(context: vscode.ExtensionContext) {
  console.log("CodePilot Free activated!");

  const provider: vscode.InlineCompletionItemProvider = {
    provideInlineCompletionItems: async (doc, pos, ctx, token) => {
      const textBeforeCursor = doc.getText(new vscode.Range(new vscode.Position(0, 0), pos));
      try {
        const res = await fetch("http://localhost:3210/v1/completions", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ prompt: textBeforeCursor }),
        });
        const data: any = await res.json();
        const completion = data?.choices?.[0]?.text || "";
        return [new vscode.InlineCompletionItem(completion, pos)];
      } catch (err) {
        console.error("CodePilot error:", err);
        return [];
      }
    },
  };

  context.subscriptions.push(
    vscode.languages.registerInlineCompletionItemProvider({ pattern: "**" }, provider)
  );
}

export function deactivate() {}
EOF

npx tsc
npx vsce package
code --install-extension codepilot-free-0.3.0.vsix || true

#######################################
# Server
#######################################
echo ">>> Setting up server..."
cd "$SERVER_DIR"
npm init -y >/dev/null
npm install express cors body-parser node-fetch@2 >/dev/null

cat > index.js <<'EOF'
const express = require("express");
const bodyParser = require("body-parser");
const cors = require("cors");
const fetch = require("node-fetch");

const app = express();
app.use(cors());
app.use(bodyParser.json());

const PORT = 3210;
const MODEL = process.env.LMSTUDIO_MODEL || "local-model";
const LMSTUDIO_URL = process.env.LMSTUDIO_URL || "http://localhost:1234/v1/completions";

app.post("/v1/completions", async (req, res) => {
  try {
    const prompt = req.body.prompt || "";
    console.log(">>> Prompt:", prompt.slice(0, 80));

    const response = await fetch(LMSTUDIO_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: MODEL,
        prompt,
        max_tokens: 200,
        temperature: 0.7,
      }),
    });

    const data = await response.json();
    res.json(data);
  } catch (err) {
    console.error("Error:", err);
    res.status(500).json({ error: err.toString() });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 CodePilot server running on http://localhost:${PORT}`);
});
EOF

#######################################
# CLI Wrapper
#######################################
echo ">>> Creating CLI launcher..."
cat > "$ROOT_DIR/codepilot.sh" <<EOF
#!/usr/bin/env bash
ROOT_DIR="$ROOT_DIR"
SERVER_DIR="\$ROOT_DIR/server"
EXT_DIR="\$ROOT_DIR/vscode-codepilot"
LOG_FILE="\$SERVER_DIR/server.log"

start_server() {
  MODEL="\${1:-local-model}"
  echo ">>> Starting server with model \$MODEL"
  pkill -f "node index.js" || true
  cd "\$SERVER_DIR"
  nohup env LMSTUDIO_MODEL="\$MODEL" node index.js > "\$LOG_FILE" 2>&1 &
  sleep 1
  echo ">>> Server started (check: codepilot status)"
}

stop_server() {
  echo ">>> Stopping server..."
  pkill -f "node index.js" || true
}

restart_server() {
  stop_server
  start_server "\$1"
}

status_server() {
  if pgrep -f "node index.js" > /dev/null; then
    echo ">>> Server is running"
  else
    echo ">>> Server is NOT running"
  fi
}

logs_server() {
  echo ">>> Tailing CodePilot logs (Ctrl+C to exit)"
  tail -f "\$LOG_FILE"
}

update_codepilot() {
  echo ">>> Updating CodePilot..."
  if [ -d "\$ROOT_DIR/.git" ]; then
    cd "\$ROOT_DIR"
    git pull --rebase
  else
    echo ">>> No Git repo, skipping pull"
  fi
  cd "\$EXT_DIR"
  npm install
  npx tsc
  npx vsce package
  code --install-extension codepilot-free-0.3.0.vsix || true
  cd "\$SERVER_DIR"
  npm install
  restart_server
}

upgrade_codepilot() {
  echo ">>> Performing full reinstall..."
  bash "\$ROOT_DIR/install_codepilot.sh"
}

ACTION="\$1"
shift || true

case "\$ACTION" in
  start) start_server "\$@" ;;
  stop) stop_server ;;
  restart) restart_server "\$@" ;;
  status) status_server ;;
  logs) logs_server ;;
  update) update_codepilot ;;
  upgrade) upgrade_codepilot ;;
  *)
    echo "Usage: codepilot {start|stop|restart|status|logs|update|upgrade} [model]"
    exit 1
    ;;
esac
EOF

chmod +x "$ROOT_DIR/codepilot.sh"
sudo ln -sf "$ROOT_DIR/codepilot.sh" "$LAUNCHER"

#######################################
# Systemd Service
#######################################
SERVICE_FILE="/etc/systemd/system/codepilot.service"
echo ">>> Creating systemd service..."
sudo bash -c "cat > $SERVICE_FILE" <<EOF
[Unit]
Description=CodePilot AI Server
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/env bash -c 'cd $SERVER_DIR && LMSTUDIO_MODEL=local-model node index.js'
Restart=always
User=$USER
Environment=PATH=$PATH

[Install]
WantedBy=default.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable codepilot
sudo systemctl start codepilot

echo
echo ">>> ✅ CodePilot installed successfully!"
echo "Commands:"
echo "    codepilot start [model]"
echo "    codepilot stop"
echo "    codepilot restart [model]"
echo "    codepilot status"
echo "    codepilot logs"
echo "    codepilot update"
echo "    codepilot upgrade"
