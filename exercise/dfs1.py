# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
# #         self.right = right
# Given the root of a binary tree and an integer targetSum, return the number of paths where the sum of the values along the path equals targetSum.

# The path does not need to start or end at the root or a leaf, but it must go downwards (i.e., traveling only from parent nodes to child nodes).
class Solution:
    def pathSum(self, root, targetSum: int) -> int:
        def dfs(root, targetSum, n, sums):
            if not root:
                return n
            for i in range(len(sums)):
                sums[i] += root.val
                if sums[i] == targetSum:
                    n += 1
                    print (root.val, n, sums)
            sums.append(root.val)
            if root.val == targetSum:
                n+=1
            n = dfs(root.left, targetSum, n, sums.copy())
            n = dfs(root.right, targetSum, n, sums.copy())
            return n
        n = dfs(root, targetSum, 0, [])
        return n