# --- Code Book: Competitive Programming
# --- Source   : github.com/sm2774us/competitive_programming_book.git

#========================================================================================================================
#       :: Arrays :: 
#       :: array/product_of_array_except_self.py ::
#       LC-238 | Product of Array Except Self | https://leetcode.com/problems/product-of-array-except-self/ | Medium
#========================================================================================================================
|  1| """
|  2| Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of nums except nums[i].
|  3|
|  4| The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
|  5|
|  6| You must write an algorithm that runs in O(n) time and without using the division operation.
|  7|
|  8| Example 1:
|  9|
| 10| Input: nums = [1,2,3,4]
| 11| Output: [24,12,8,6]
| 12| """
| 13| 
| 14| # Optimized Space Solution. Without using extra memory of left and right product list.
| 15| # O(N)
| 16| # SC: O(1) [ excluding the output/result array, which does not count towards extra space, as per problem description. ]
| 17| from typing import List
| 18| def productExceptSelf(nums: List[int]) -> List[int]:
| 20|     length_of_list = len(nums)
| 21|     result = [0]*length_of_list
| 22|
| 23|     # update result with left product.
| 24|     result[0] = 1
| 25|     for i in range(1, length_of_list):
| 26|         result[i] = result[i-1] * nums[i-1]
| 27|
| 28|     right_product = 1
| 29|     for i in reversed(range(length_of_list)):
| 30|         result[i] = result[i] * right_product
| 31|         right_product *= nums[i]
| 32|
| 33|     return result

#====================================================================================================================================================
#       :: Arrays :: 
#       :: array/subarray_sum_equals_k.py ::
#       LC-560 | Subarray Sum Equals K | https://leetcode.com/problems/subarray-sum-equals-k/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given an array of integers nums and an integer k, return the total number of continuous subarrays whose sum equals to k.
|  3|
|  4| Example 1:
|  5|
|  6|   Input: nums = [1,1,1], k = 2
|  7|   Output: 2
|  8| """
|  9| 
| 10| #Logic prefix sums and HashMap ( i.e., dictionary )
| 11| #
| 12| #TC: O(N)
| 13| #SC: O(N)
| 14| import collections
| 15| from typing import List
| 16| def subarraySum(nums: List[int], k: int) -> int:
| 17|     prefix_sums = coolections.defaultdict(int)
| 18|     curr_sum = 0
| 19|     res = 0
| 20|     for i in range(len(nums)):
| 21|         curr_sum += nums[i]
| 22|         if curr_sum == k:
| 23|             res += 1
| 24|         if (curr_sum - k) in prefix_sums:
| 25|             res += prefix_sums[curr_sum - k]  # this is the number of contiguous subarrays between prev curr_sum and current one that have sum equal to k
| 26|         prefix_sums[curr_sum] += 1
| 27|     return res

#====================================================================================================================================================
#       :: Arrays :: 
#       :: array/subarray_sum_equals_k_facebook_i_variant.py ::
#       Facebook Interview - I (Variant of: LC-560 - Subarray Sum Equals K ) | Facebook - I - Variant of Subarray Sum Equals K | Medium
#       https://leetcode.com/discuss/interview-question/algorithms/124820/continuous-sequence-against-target-number
#       [ Only consider positive numbers, so we can use Sliding Window to improve Space Complexity from O(N) to O(1) ]
#====================================================================================================================================================
|  1| """
|  2| Given a sequence of positive integers nums and an integer k, return whether there is a continuous sequence of nums that sums up to exactly k.
|  3|
|  4| Example 1:
|  5|
|  6|   Input: nums = [23, 5, 4, 7, 2, 11], k = 20
|  7|   Output: true
|  8|   Output: 7 + 2 + 11 = 20
|  9|
| 10| Example 2:
| 11|
| 12|   Input: nums = [1, 3, 5, 23, 2], k = 8
| 13|   Output: true
| 14|   Explanation: 3 + 5 = 8
| 15|
| 16| Example 3:
| 17|
| 18|   Input: nums = [1, 3, 5, 23, 2], k = 7
| 19|   Output: false
| 20|   Explanation: because no sequence in this array adds up to 7.
| 21| """
| 22| 
| 23| #Logic sliding window keep opening window as long as sum is less than target
| 24| #start closing window as long as sum exceeds target 
| 25| #
| 26| #TC: O(N)
| 27| #SC: O(1)
| 28| def sum_array_exists(a, target):
| 29|     n = len(a)
| 30|     begin = 0
| 31|     end    = 0
| 32|     xsum = 0
| 33|     while end < n:
| 34|         if xsum == target:
| 35|             return True
| 36|         if xsum < target:
| 37|             end += 1  # open window to include more items
| 38|         if xsum > target:
| 39|             xsum -= a[begin]
| 40|             begin += 1 # start closing window
| 41|     return False

#=======================================================================================================================================================
#       :: Arrays :: 
#       :: array/subarray_sum_equals_k_non_overlapping_intervals_facebook_ii_variant.py ::
#       Facebook Interview - II (Variant of: LC-560 - Subarray Sum Equals K ) | Facebook - Max Sum of Non-overlapping Subarray with Fixed Sum | Medium
#       https://leetcode.com/discuss/interview-question/750317/facebook-phone-max-sum-of-non-overlapping-subarray-with-fixed-sum
#======================================================================================================================================================
|  1| """
|  2| This question contains two part:
|  3|
|  4|   1. Count the number of subarrays (continuous) whose sum equal to the given k.
|  5|   2. In all subarrays found above, find as many non-overlapping subarrays as possible that gives the maximal total sum. Output the total sum.
|  6|
|  7| Example:
|  8|
|  9|   [1,1,1], k = 2, there are 2 subarrays whose sum is 2, but they are overlapping.
| 10|   So, output 2 instead of 4.
| 11|
| 12| """
| 13| 
| 14| #Logic prefix sums and HashMap ( i.e., dictionary )
| 15| #Algorithm Steps:
| 16| #1. Iterate over the list.
| 17| #   1.1. Copy values of prefix_sums into a result array and reset prefix_sums,
| 18| #        each time we encounter a match ( curr_sum - k ). 
| 19| #2. The result array will have the non-overlapping subarrays.
| 20| #3. Sum up the values in the result array and return it. 
| 21| #
| 22| #TC: O(N)
| 23| #SC: O(N)
| 24| import collections
| 25| from typing import List
| 26| def subArraySumNonOverlappingSubArrays(nums: List[int], k: int) -> int:
| 27|     prefix_sums = collections.defaultdict(int)
| 28|     prefix_sums[0] = [-1]
| 31|     curr_sum = 0
| 32|     res = []
| 33|     for i in range(len(nums)):
| 34|         curr_sum += nums[i]
| 35|         if (curr_sum - k) in prefix_sums:
| 36|             res.append(*[nums[value+1:i+1] for value in prefix_sums[curr_sum-k]])
| 37|             prefix_sums.clear()
| 38|             prefix_sums[0] = []
| 39|         else:
| 40|             prefix_sums[curr_sum].append(i)
| 41|     return sum(list(map(sum, res)))

#====================================================================================================================================================
#       :: Arrays :: 
#       :: array/maximum_subarray.py ::
#       LC-53 | Maximum Subarray | https://leetcode.com/problems/maximum-subarray/ | Easy
#====================================================================================================================================================
|  1| """
|  2| Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
|  3|
|  4| Example 1:
|  5|
|  6|   Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
|  7|   Output: 6
|  8|   Explanation: [4,-1,2,1] has the largest sum = 6.
|  9| """
| 10| 
| 11| """
| 12| Kadane's Algorithm
| 13| ------------------------------------------------------------------------------------------
| 14| 1. The largest subarray is either:
| 15|   - the current element
| 16|   - sum of previous elements
| 17|
| 18| 2. If the current element does not increase the value a local maximum subarray is found.
| 19|
| 20| 3. If local > global replace otherwise keep going.
| 21| ------------------------------------------------------------------------------------------
| 22| """
| 23|
| 24| # TC: O(N)
| 25| # The time complexity of Kadane’s algorithm is O(N) because there is only one for loop which scans the entire array exactly once.
| 26|
| 27| # SC: O(1)
| 28| # Kadane’s algorithm uses a constant space. So, the space complexity is O(1).
| 29| from typing import List
| 30| def maxSubArray(self, nums: List[int]) -> int:
| 31|     maxGlobals = nums[0]  # largest subarray for entire problem
| 32|     maxCurrent = nums[0]  # current max subarray that is reset
| 33|     for i in range(1,len(nums)):
| 34|         maxCurrent = max(nums[i], maxCurrent+nums[i])
| 35|         maxGlobal = max(maxCurrent, maxGlobal)
| 36|     return maxGlobal

#====================================================================================================================================================
#       :: Arrays :: 
#       :: array/maximum_subarray_variant.py ::
#       Maximum Subarray - Return Subarray (Variant of: LC-53 - Maximum Subarray ) | Maximum Subarray - Return Subarray | Medium
#====================================================================================================================================================
|  1| """
|  2| Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
|  3|
|  4| Example 1:
|  5|
|  6|   Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
|  7|   Output: [4,-1,2,1]
|  8|   Explanation: [4,-1,2,1] has the largest sum = 6.
|  9| """
| 10|
| 11| #TC: O(N)
| 12| #SC: O(N) - because we have to return the subarray which in the worst case can be N.
| 13| from typing import List
| 14| def maxSubArray(A: List[int]) -> List[int]:
| 15|     start = end = 0              # stores the starting and ending indices of max sublist found so far
| 16|     beg = 0                      # stores the starting index of a positive-sum sequence
| 17|     for i in range(len(A)):
| 18|         maxEndingHere = maxEndingHere + A[i]
| 19|         if maxEndingHere < 0:
| 20|             maxEndingHere = 0
| 21|             beg = i + 1
| 22|         if maxSoFar < maxEndingHere:
| 23|             maxSoFar = maxEndingHere
| 24|             start = beg
| 25|             end = i
| 26|     return A[start:end+1]
