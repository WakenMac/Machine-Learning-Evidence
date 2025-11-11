


class QueuedList:
    """
    A one-way linked list with a soecific queue and a specific size.
    """

    def __init__(self, max_size:int = 1):
        self.max_size = max_size if max_size > 0 else 10
        self.current_size = 0
        self.start_node = None
        self.end_none = None

    def append(self, value:float):
        new_node = Node(value)

        if self.current_size == 0:
            self.start_node = new_node
            self.end_node = new_node
            self.current_size += 1
        
        elif self.current_size < self.max_size:
            self.end_node.next = new_node
            self.end_node = self.end_node.next
            self.current_size += 1

        else:
            self.start_node = self.start_node.next
            self.end_node.next = new_node
            self.end_node = self.end_node.next

        return True
    
    def get_sum(self) -> float:
        sum = 0
        temp_node = self.start_node
        while temp_node:
            sum += temp_node.value
            temp_node = temp_node.next
        return sum
    
    def get_mean(self) -> float:
        sum = self.get_sum()
        return sum / self.current_size
    
    def __str__(self):
        values = []
        temp = self.start_node
        while temp:
            values.append(str(temp.value))
            temp = temp.next
        return " -> ".join(values)

class Node:

    def __init__(self, value:float = 0):
        self.value = value if value is not None else 0
        self.next = None