from typing import List, Tuple


def main1(a: List[int]) -> int:
    max_sum = 0
    curr_sum = 0
    for i in a:
        if curr_sum + i < 0:
            max_sum = curr_sum
            curr_sum = 0
    if curr_sum > max_sum:
        max_sum = curr_sum

    return max_sum

def print_matrix(a: List[List[int]]) -> None:
    for row in a:
        print(" ".join([str(i) for i in row]))
    print("\n")

def valid_neighbors(a: List[List[int]], point: Tuple[int,int]) -> List[Tuple[int,int]]:
    r, c = point
    neighbors = []
    if r > 0:
        neighbors.append((r-1,c))
    if r < len(a)-1:
        neighbors.append((r+1,c))
    if c > 0:
        neighbors.append((r,c-1))
    if c < len(a[0])-1:
        neighbors.append((r,c+1))
    return neighbors

def main2(a: List[List[int]], t: Tuple[int, int]) -> List[Tuple[int, int]]:
    
    lands = [[0 for i in range(len(a[0]))] for j in range(len(a))]
    candidates = [t]
    while candidates:
        candidate = candidates.pop()
        lands[candidate[0]][candidate[1]] = 1
        for neighbor in valid_neighbors(a, candidate):
            if lands[neighbor[0]][neighbor[1]] == 1:
                continue
            if a[neighbor[0]][neighbor[1]] == 1:
                candidates.append(neighbor)



    print_matrix(a)
    print_matrix(lands)


# a = [[0, 1, 1, 0, 0, 0, 0],
#      [0, 0, 0, 0, 0, 0, 0],
#      [0, 1, 1, 1, 1, 1, 1],
#      [0, 1, 0, 0, 1, 0, 1],
#      [0, 1, 1, 1, 1, 1, 1]]

# main2(a, (2, 1))

# You have a list of cities and populations. Return a random city but the random has to be weighed by the population. So you build array of cumulative weights and then random number. But the difficulty is you need to map that back to the right city. O(n) time would search for city with the number in the right bracket but O(Log N) would binary search the brackets. 


# Coinchange

# You have a list of cities and populations. Return a random city but the random has to be weighed by the population. So you build array of cumulative weights and then random number. But the difficulty is you need to map that back to the right city. O(n) time would search for city with the number in the right bracket but O(Log N) would binary search the brackets. 



# Given an array of ints, return the k most frequently occurring ints. Either use a heap or make a histogram, then sort on the values, then take top k.
# def main3(a: List[int], k: int) -> List[int]:


#     c = Counter(a)
#     return nlargest(k, c, key=c.get)

# add two decimal numbers
def main4(a: str, b: str) -> float:
    a1, a2 = a.split(".")
    b1, b2 = b.split(".")

    num_digits = max(len(a1), len(b1))
    sum = 0
    for a in range(num_digits):
        power = 10 ** a
        a_digit = int(a1[-1-a]) if a < len(a1) else 0
        b_digit = int(b1[-1-a]) if a < len(b1) else 0
        sum += power * (a_digit + b_digit)
    
    num_digits = max(len(a1), len(b1))
    for a in range(num_digits):
        power = 10 ** (-(a+1))
        a_digit = int(a2[a]) if a < len(a2) else 0
        b_digit = int(b2[a]) if a < len(b2) else 0
        sum += power * (a_digit + b_digit)
        
    return sum

def is_palindrome(s: str) -> bool:
    a = 0
    b = len(s) - 1
    while a < b:
        if not s[a].isalpha():
            a += 1
        if not s[b].isalpha():
            b -= 1
        if str.lower(s[a]) != str.lower(s[b]):
            return False
        a += 1
        b -= 1
    return True


#2 (this is the one I came only with brute force)
Given a pile of sticks and array representing sticks lengths, create a function that for given K will return max length of the stick that can be cut the way there is at least K sticks of this length.
You can cut sticks in how many pieces you want, no glueing.
[5,7,9], k=3 -> 5
[5,7,9], k=4 -> 4
[30,5,7,9], k=4 -> 9