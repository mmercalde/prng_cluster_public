#!/usr/bin/env python3
with open('coordinator.py', 'r') as f:
    content = f.read()

old = '''                        class SieveResult:
                            def __init__(self, d):
                                self.success = d.get('success', True)
                                self.runtime = d.get('runtime', 0)
                                self.error = d.get('error', None)
                                self.survivors_found = d.get('survivors_found', 0)
                                self.__dict__.update(d)'''

new = '''                        class SieveResult:
                            def __init__(self, d):
                                self.success = d.get('success', True)
                                self.runtime = d.get('runtime', 0)
                                self.error = d.get('error', None)
                                self.survivors_found = d.get('survivors_found', 0)
                                self.results = d.get('results', d)  # Support both formats
                                self.__dict__.update(d)'''

content = content.replace(old, new)
with open('coordinator.py', 'w') as f:
    f.write(content)
print("✅ Fixed!")
