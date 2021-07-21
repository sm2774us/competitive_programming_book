#====================================================================================================================================================
#       :: Arrays ::
#       :: Prefix Sum + Modular Arithmetic ::
#       :: arrays/continuous_subarray_sum.py ::
#       LC-523 | Continuous Subarray Sum | https://leetcode.com/problems/continuous-subarray-sum/ | Medium
#====================================================================================================================================================
#"""
#Given an integer array nums and an integer k, return true if nums has a continuous subarray of size at least two whose elements 
#sum up to a multiple of k, or false otherwise.
#
#An integer x is a multiple of k if there exists an integer n such that x = n * k. 0 is always a multiple of k.
#
#Example 1:
#
#  Input: nums = [23,2,4,6,7], k = 6
#  Output: true
#  Explanation: [2, 4] is a continuous subarray of size 2 whose elements sum up to 6.
#
#Example 2:
#
#  Input: nums = [23,2,6,4,7], k = 6
#  Output: true
#  Explanation: [23, 2, 6, 4, 7] is an continuous subarray of size 5 whose elements sum up to 42.
#  42 is a multiple of 6 because 42 = 7 * 6 and 7 is an integer.
#
#Example 3:
#
#  Input: nums = [23,2,6,4,7], k = 13
#  Output: false
#"""
#
#"""
#Prefix Sum Review
#---------------------------------
#We can solve this question with a prefix sum, an array where each position is the accumulated sum until the current index, but not including it.
#
#For example:
#
#  * arr = [23, 2, 4, 6, 7]
#  * prefix_sum = [0, 23, 25, 29, 35, 42]
#
#And, with prefix sum, we can also query the contiguous subarray sum between two index. Following the same example:
#
#  prefix_sum = [0, 23, 25, 29, 35, 42]
#  # Sum between element 1 and 3: (2, 4, 6)
#  prefix_sum[3 + 1] - prefix_sum[1] = 35 - 23 = 12
#
#  * prefix_sum[3 + 1] because we want to include the element at index 3, so : (23 + 2 + 4 + 6 + 7).
#  * prefix_sum[1] because we want to remove all elements until the element at idx 1: (23)
#  * prefix_sum[4] - prefix_sum[1] = (23 + 2 + 4 + 6 + 7) - 23 = 2 + 4 + 6 + 7
#
#Question's Idea
#---------------------------------
#Now we know more about the prefix_sum, lets think what is have a sum as multiple of k. Given the example nums = [23, 2, 4, 6, 7], k = 6, all valid sums are:
#
#  * (2 + 4) = 6
#  * (6) = 6
#  * (23 + 2 + 4 + 6 + 7) = 42
#
#  We also can use a modulo property, where: 
#
#  (A + B) % K = (A % K + B % K) % K
#
#  So we can use a modulo in our accumulated array and, instead of looking for a sum equals to k*n, we want to find a sum equals to k.
#
#Summarizing everything:
#---------------------------------
#arr = [23, 2, 4, 6, 7]
#prefix_sum = [0, 23, 25, 29, 35, 42]
#modulo_prefix_sum = [0, 5, 1, 5, 5, 0]
#
#But wait! Now we don't have a valid prefix_sum, since it isn't crescent.
#Yes, and that's why we want to look forward similar values. For example, we have two 5's. That means that, in my prefix_sum, 
#I had two values where sum % k = 5. But what are we looking for now? Let's think:
#
# A + B = n * k # simbolizes that A + B are multiple of k (k * n is equal to that sum)
# (A + B) % k = (n * k) % k
# (A % k + B % k) % k = n % k * k % k
# (A % k + B % k) % k = 0 # (OMG!!)
#
#Using the previous idea, we know now that we have to find a subarray where (nums[i] % k + nums[i+1] % k + ... nums[j] % k) % k == 0.
#But, knowing our prefix_sum, we know that prefix_sum[j + 1] - prefix_sum[i] is exactly that sum!!!.
#So, using everything we already learned, prefix_sum[j + 1] == prefix_sum[i]!!!!!!!!!!
#
#Conclusion:
#---------------------------------
#So, thanks to the explanation before, we are looking for equal values in our custom prefix_sum (prefix_sum with modulos) 
#and i and j cannot be consecutives (if j = i + 1, prefix_sum[i + 1] - prefix_sum[i] means the sum between [i, i], a unique value)
#"""
#
#Prefix Sum + Modular Arithmetic Solution
#TC: O(N)
#SC: O(N)
import itertools
from typing import List
def checkSubarraySum(nums: List[int], k: int) -> bool:
    # We don't want to do (num % 0,) right? hehe
    if not k:
        # If k is 0 and there is no negative value, the only possible solution is a subarray like [0,0] (e.g. [1,7,3,0,0,1,2,5,0,1])
        return any([nums[i] + nums[i - 1] == 0 for i in range(1, len(nums))])
       
    prefix_sum = [0 for _ in range(len(nums) + 1)]

    for idx in range(len(nums)):
        prefix_sum[idx + 1] = (prefix_sum[idx] + nums[idx] % k) % k
           
    memo = {}

    for idx, curr_sum in enumerate(prefix_sum):
        if curr_sum in memo and idx - memo[curr_sum] > 1:
            return True

        if curr_sum not in memo:
            memo[curr_sum] = idx

    return False

# More concise soluton using python builtin itertools.accumulate ( Good to know but don't use in an interview setting )
def checkSubarraySumConcise(nums: List[int], k: int) -> bool:
    mapping = {0: -1}
    for i, prefix_sum in enumerate(itertools.accumulate(nums)):
        key = prefix_sum % k if k else prefix_sum
        if key not in mapping:
            mapping[key] = i
        elif i - mapping[key] > 1:
            return True
    return False
