#====================================================================================================================================================
#       :: Arrays ::
#       :: Kadane's Algorithm ::
#       :: arrays/maximum_subarray_variant.py ::
#       Maximum Subarray - Return Subarray (Variant of: LC-53 - Maximum Subarray ) | Maximum Subarray - Return Subarray | Medium
#====================================================================================================================================================
#"""
#Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
#
#Example 1:
#
#  Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
#  Output: [4,-1,2,1]
#  Explanation: [4,-1,2,1] has the largest sum = 6.
#"""
#
#TC: O(N)
#SC: O(N) - because we have to return the subarray which in the worst case can be N.
from typing import List
def maxSubArray(A: List[int]) -> List[int]:
    start = end = 0              # stores the starting and ending indices of max sublist found so far
    beg = 0                      # stores the starting index of a positive-sum sequence
    for i in range(len(A)):
        maxEndingHere = maxEndingHere + A[i]
        if maxEndingHere < 0:
            maxEndingHere = 0
            beg = i + 1
        if maxSoFar < maxEndingHere:
            maxSoFar = maxEndingHere
            start = beg
            end = i
    return A[start:end+1]
