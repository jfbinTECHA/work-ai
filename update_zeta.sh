 
#!/bin/bash
# update_zeta.sh
# Patch script for Nomi AI project (Zeta system update)

set -e

echo "=== Updating Zeta system for Nomi AI ==="

# --- 1. Update zeta.py ---
if [ -f "zeta.py" ]; then
  cp zeta.py zeta.py.bak
  echo "Backup of zeta.py saved as zeta.py.bak"

  # Replace text encoding scheme line
  sed -i 's|.*text_encoding_scheme.*|text_encoding_scheme = lambda text: [ord(char) for char in text]|' zeta.py

  echo "zeta.py updated."
else
  echo "zeta.py not found!"
fi

# --- 2. Update predictor.py ---
if [ -f "predictor.py" ]; then
  cp predictor.py predictor.py.bak
  echo "Backup of predictor.py saved as predictor.py.bak"

  # Insert AttentionLayer before predict definition
  awk '
  /def predict/ && !done {
      print "class AttentionLayer(nn.Module):"
      print "    def __init__(self, embed_dim, hidden_dim):"
      print "        super().__init__()"
      print "        self.W = nn.Linear(embed_dim, hidden_dim)"
      print "        self.U = nn.Linear(hidden_dim, hidden_dim)"
      print "        self.hidden_dim = hidden_dim"
      print ""
      print "    def forward(self, query, keys):"
      print "        query_hidden = self.W(query)"
      print "        keys_hidden = self.U(keys)"
      print "        weights = torch.matmul(query_hidden, keys_hidden.T) / math.sqrt(self.hidden_dim)"
      print "        return torch.softmax(weights, dim=1)"
      print ""
      done=1
  }
  {print}
  ' predictor.py > predictor.py.tmp && mv predictor.py.tmp predictor.py

  # Ensure imports exist
  grep -q "import torch" predictor.py || sed -i '1i import torch' predictor.py
  grep -q "import torch.nn as nn" predictor.py || sed -i '2i import torch.nn as nn' predictor.py
  grep -q "import math" predictor.py || sed -i '3i import math' predictor.py

  echo "predictor.py updated."
else
  echo "predictor.py not found!"
fi

# --- 3. Update ai_zeta/models/model.py ---
if [ -f "ai_zeta/models/model.py" ]; then
  cp ai_zeta/models/model.py ai_zeta/models/model.py.bak
  echo "Backup of model.py saved as model.py.bak"

  # Replace calculate_reward function
  sed -i '/def calculate_reward/,/^$/c\def calculate_reward(action, outcome):\n    return action * outcome\n' ai_zeta/models/model.py

  echo "model.py updated."
else
  echo "ai_zeta/models/model.py not found!"
fi

echo "=== Update complete! ==="
