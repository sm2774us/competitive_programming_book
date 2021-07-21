#====================================================================================================================================================
#       :: Arrays ::
#       :: Kadane's Algorithm :: 
#       :: arrays/maximum_subarray.py ::
#       LC-53 | Maximum Subarray | https://leetcode.com/problems/maximum-subarrays/ | Easy
#====================================================================================================================================================
#Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
#
#Example 1:
#
#  Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
#  Output: 6
#  Explanation: [4,-1,2,1] has the largest sum = 6.
#
#
#Kadane's Algorithm
#------------------------------------------------------------------------------------------
#1. The largest subarray is either:
#  - the current element
#  - sum of previous elements
#
#2. If the current element does not increase the value a local maximum subarray is found.
#
#3. If local > global replace otherwise keep going.
#------------------------------------------------------------------------------------------
#
# TC: O(N)
# The time complexity of Kadane’s algorithm is O(N) because there is only one for loop which scans the entire array exactly once.
#
# SC: O(1)
# Kadane’s algorithm uses a constant space. So, the space complexity is O(1).
from typing import List
def maxSubArray(nums: List[int]) -> int:
    maxGlobals = nums[0]  # largest subarray for entire problem
    maxCurrent = nums[0]  # current max subarray that is reset
    for i in range(1,len(nums)):
        maxCurrent = max(nums[i], maxCurrent+nums[i])
        maxGlobal = max(maxCurrent, maxGlobal)
    return maxGlobal
