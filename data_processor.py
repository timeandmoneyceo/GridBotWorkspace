class DataProcessor:
    """Process data with various methods"""
    
    def __init__(self):
        self.data = []
    
    def add_data(self, item):
        """Add item to data"""
        self.data.append(item)
    
    def process_data(self):
        """Process the data"""
        return [item * 2 for item in self.data if isinstance(item, (int, float))]