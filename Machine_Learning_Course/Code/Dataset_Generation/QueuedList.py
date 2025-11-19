# Author: Waken Cean C. Maclang
# Date Last Edited: November 11, 2025
# Course: Machine Learning
# Task: Learning Evidence

# QueuedList.py 
#     Contains the QueuedList & Node class.
#     It is the main data structure for calculating velocity (change in 
#        distance and displacement) of the hand given a specific fps  

# Works with Python 3.10.0


class QueuedList:
    """
    A one-way linked list with a specific queue and a specific size.
    NOTE: Designed for numericals due to having sum and mean functions.

    It allows for dynamic adding of a new value to the end of the list, 
       while not exceeding its maximum size by shifting the list to the 
       last remaining or recently added elements.

    Imagine it as a queue with a line limit, where after the line is full 
       and a new customer apprears, the most recent in line leaves, every
       customer in line shifts once to the front, and the new customer 
       fills the gap at the end of the line.

    The structure also allows for sum and mean aggregation of values.
    
    Exmaple:

    ql = QueuedList(3)   # max_size of our QueuedList is 3
    ql.append(1)
    ql.append(2)
    ql.append(3)
    print(ql)            # (Front) 1 -> 2 -> 3 (End)

    ql.append(4)       
    print(ql)            # (Front) 2 -> 3 -> 4 (End)

    print(ql.get_sum())  # 9

    ql.append(5)
    print(ql.get_sum())  # 13
    print(ql.get_mean()) # 4.333...3
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