import os
import json
from datetime import datetime

class AgentLogger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, sample_id, trace, final_output):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "trace": trace,
            "final_output": final_output
        }
        file_path = os.path.join(self.log_dir, f"{sample_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)
