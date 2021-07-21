#========================================================================================================================
#       :: Arrays :: 
#       :: arrays/product_of_array_except_self.py ::
#       LC-238 | Product of Array Except Self | https://leetcode.com/problems/product-of-array-except-self/ | Medium
#========================================================================================================================
#"""
#Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
#
#The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
#
#You must write an algorithm that runs in O(n) time and without using the division operation.
#
#Example 1:
#
#Input: nums = [1,2,3,4]
#Output: [24,12,8,6]
#"""
#
# Optimized Space Solution. Without using extra memory of left and right product list.
# TC: O(N)
# SC: O(1) [ excluding the output/result array, which does not count towards extra space, as per problem description. ]
from typing import List
def productExceptSelf(nums: List[int]) -> List[int]:
    length_of_list = len(nums)
    result = [0]*length_of_list

    # update result with left product.
    result[0] = 1
    for i in range(1, length_of_list):
        result[i] = result[i-1] * nums[i-1]

    right_product = 1
    for i in reversed(range(length_of_list)):
        result[i] = result[i] * right_product
        right_product *= nums[i]

    return result
