import yaml
import csv

class Memory:
    def __init__(self, criteria_path, glossary_path):
        self.criteria = self.load_criteria(criteria_path)
        self.glossary = self.load_glossary(glossary_path)
        self.temp_memory = {}

    def load_criteria(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def load_glossary(self, path):
        glossary = {}
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                glossary[row['english']] = row['filipino']
        return glossary

    def set_temp(self, key, value):
        self.temp_memory[key] = value

    def get_temp(self, key):
        return self.temp_memory.get(key, None)

    def clear_temp(self):
        self.temp_memory.clear()
