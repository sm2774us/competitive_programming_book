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
| 30| def maxSubArray(nums: List[int]) -> int:
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

#====================================================================================================================================================
#       :: Arrays :: 
#       :: array/find_the_duplicate_number.py ::
#       LC-287 | Find the Duplicate Number | https://leetcode.com/problems/find-the-duplicate-number/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given an array of integers nums containing n + 1 integers where each integer is in the range [1, n] inclusive.
|  3|
|  4| There is only one repeated number in nums, return this repeated number.
|  5|
|  6| You must solve the problem without modifying the array nums and uses only constant extra space.
|  7|
|  8| Example 1:
|  9|
| 10|   Input: nums = [1,3,4,2,2]
| 11|   Output: 2
| 12|
| 13| Example 2:
| 14|
| 15|   Input: nums = [3,1,3,4,2]
| 16|   Output: 3
| 17|
| 18| Example 3:
| 19|
| 20|   Input: nums = [1,1]
| 21|   Output: 1
| 22|
| 23| Example 4:
| 24|
| 25|   Input: nums = [1,1,2]
| 26|   Output: 1
| 27|
| 28|
| 29| Constraints:
| 30|   * 1 <= n <= 105
| 31|   * nums.length == n + 1
| 32|   * 1 <= nums[i] <= n
| 33|   * All the integers in nums appear only once except for precisely one integer which appears two or more times.
| 34| """
| 35|
| 36| """
| 37| One way to handle this problem is to sort data with O(n log n) time and O(n) memory or use hash table.
| 38| Howerer it violates the conditions of problem: we can not use extra memory.
| 39| There is smarter way and we need to use the fact that each integer is between 1 and n, which we did not use in sort.
| 40|
| 41| Let us deal with our list as a linked list, where i is connected with nums[i].
| 42| Consider example 6, 2, 4, 1, 3, 2, 5, 2. Then we have the following singly-linked list:
| 43| 0 -> 6 -> 5 -> 2 -> 4 -> 3 -> 1 -> 2 -> ...
| 44| We start with index 0, look what is inside? it is number 6, so we look at index number 6, what is inside? Number 5 and so on. 
| 45| Look at the image below for better understanding.
| 46| So the goal is to find loop in this linkes list. Why there will be always loop? 
| 47| Because nums[1] = nums[5] in our case, and similarly there will be always duplicate, and it is given that it is only one.
| 48|    +-------------------------------------+
| 49|    |               __________            |
| 50|    |        ______/          \           |
| 51|    |       /     / \   ___    \    ___   |
| 52|    |      / +***/*+ \ /   \    \  /   \  |
| 53|  +-+-+ +-▼-+*+-▼-+*+-▼-+ +-+-+ +-▼-+ +-+-▼ +-+-+
| 54|  | 6 | | 2 |*| 4 |*| 1 | | 3 | | 2 | | 5 | | 2 |
| 55|  +---+ +-+-+*+-▲-+*+---+ +-▲-+ +---+ +---+ +---+
| 56|           \_*__/ \*________/
| 57|             +*****+
| 58|                ▲
| 59|                |_____ Start of Loop
| 60|               ===
| 61|    0     1   = 2 =   3     4     5     6     7
| 62|              =   =     
| 63|               ===
| 64|
| 65|    0  ➜ 6  ➜  5  ➜ 2  ➜  4  ➜ 3  ➜  1
| 66|                      ▲                /
| 67|                       \______________/
| 68|
| 69| So now, the problem is to find the starting point of loop in singly-linked list (problem 142),
| 70| which has a classical solution with two pointers: slow which moves one step at a time and fast, 
| 71| which moves two times at a time ( Floyd's Cycle Detection Algorithm ). To find this place 
| 72| we need to do it in two iterations: first we wait until fast pointer gains slow pointer 
| 73| and then we move slow pointer to the start and run them with the same speed and wait until they concide.
| 74|
| 75| Complexity: Time complexity is O(n), because we potentially can traverse all list.
| 76| Space complexity is O(1), because we actually do not use any extra space: our linked list is virtual.
| 77| """
| 78|
| 79| #Approach: Linked List Cycle (Floyd's Cycle Detection Algorithm)
| 80| #TC: O(N), because we potentially can traverse all list
| 81| #SC: O(1), because we actually do not use any extra space: our linked list is virtual
| 82| from typing import List
| 83| def findDuplicate(nums: List[int]) -> int:
| 84|     slow, fast = nums[0], nums[0]
| 85|     while True:
| 86|         slow, fast = nums[slow], nums[nums[fast]]
| 87|         if slow == fast: break
| 88| 
| 89|     slow = nums[0]
| 90|     while slow != fast:
| 91|         slow, fast = nums[slow], nums[fast]
| 92|     return slow

#====================================================================================================================================================
#       :: Arrays :: 
#       :: array/remove_duplicates_from_unsorted_array.py ::
#       Google Interview | Easy
#       https://leetcode.com/discuss/interview-question/168757/Google%3A-Remove-Duplicates-from-Unsorted-Array
#====================================================================================================================================================
|  1| """
|  2| Problem Requirements:
|  3| * can't use Set
|  4| * array is not sorted
|  5| * must be done in-place
|  6| """
|  7|
|  8| """
|  9| * Maintain a dict seen for all the numbers we have encountered before
| 10| * Handle the two cases:
| 11|   + Case 1: If not in seen then keep in the input array
| 12|   + Case 2: If in seen then delete from the input array
| 13| """
| 14| 
| 15| # Solution-1: Use this since the problem specifies that we can't use Set. 
| 16| #TC: O(N)
| 17| #SC: O(N)
| 18| def del_dups(seq: List[int]) -> None:
| 19|     seen = {}
| 20|     pos = 0
| 21|     for item in seq:
| 22|         if item not in seen:
| 23|             seen[item] = True
| 24|             seq[pos] = item
| 25|             pos += 1
| 26|     del seq[pos:]
| 27|
| 28| # Soluton-2: If we need to preserve the input array/list
| 29| #TC: O(N)
| 30| #SC: O(N)
| 31| def del_dups(seq: List[int]) -> List[int]:
| 32|     seen = {}
| 33|     pos = 0
| 34| 	  result = []
| 35|     for item in seq:
| 36|         if item not in seen:
| 37|             seen[item] = True
| 38|             seq[pos] = item
| 39|             pos += 1
| 40| 			result.append(item)
| 41|     return result
| 42|
| 43| def del_dups_set(seq: List[int]) -> None:
| 44|     seen = set()
| 45|     seen_add = seen.add
| 46|     pos = 0
| 47|     for item in seq:
| 48|         if item not in seen:
| 49|             seen_add(item)
| 50|             seq[pos] = item
| 51|             pos += 1
| 52|     del seq[pos:]
| 53|  
| 54| def del_dups_fromkeys(seq: List[int]) -> None:
| 55| 	seq[:] = dict.fromkeys(seq)
| 56|  
| 57| lst = [8, 8, 9, 9, 7, 15, 15, 2, 20, 13, 2, 24, 6, 11, 7, 12, 4, 10, 18,
| 58|        13, 23, 11, 3, 11, 12, 10, 4, 5, 4, 22, 6, 3, 19, 14, 21, 11, 1,
| 59|        5, 14, 8, 0, 1, 16, 5, 10, 13, 17, 1, 16, 17, 12, 6, 10, 0, 3, 9,
| 60|        9, 3, 7, 7, 6, 6, 7, 5, 14, 18, 12, 19, 2, 8, 9, 0, 8, 4, 5]
| 61|  
| 62| def measure(func):
| 63|     from timeit import Timer
| 64|  
| 65|     N = 50000
| 66|     setup = 'from __main__ import lst, del_dups, del_dups_set, del_dups_fromkeys'
| 67|     t = Timer('%s(lst[:])' % (func,), setup)
| 68|     print("%s: %.2f us" % (func, 1e6*min(t.repeat(number=N))/N,))
| 69| 
| 70| if __name__=="__main__":
| 71|     measure('del_dups')
| 72|     measure('del_dups_set')
| 73|     measure('del_dups_fromkeys')

#====================================================================================================================================================
#       :: Arrays :: 
#       :: array/remove_duplicates_from_sorted_array.py ::
#       Google Interview | Easy
#       https://leetcode.com/discuss/interview-question/168757/Google%3A-Remove-Duplicates-from-Unsorted-Array
#====================================================================================================================================================
|  1| """
|  2| Now we can make use of Binary Search.
|  3| """
|  4|`
|  5| # Binary search function
|  4| # returns the index of the first occurence of
|  5| # element ele in list_to_search between index
|  6| # left and right
|  7| # @param list_to_search : list in which to search the element ele
|  8| # @param left : left index for where to search element in list
|  9| # @param right : right index for where to search element in list
| 10| # @param ele : element to search in list
| 11| def binSearch(list_to_search, left, right, ele): 
| 12|     # check if the lft index is always smaller
| 13|     # than right index
| 14|     if right >= left: 
| 15|         # calculate the middle index
| 16|         middle = left + (right - left) // 2
| 17|         # check if the middle index of the list
| 18|         # consists the element, then return
| 19|         # the middle element
| 20|         if list_to_search[middle] == ele: 
| 21|             return middle 
| 22|         # Check if the element is less than element at middle
| 23|         # Do Binary Search on left sub array because only
| 24|         # left sub array will contain the element
| 25|         elif list_to_search[middle] > ele: 
| 26|             return binSearch(list_to_search, left, middle-1, ele) 
| 27|         # Check if the element is greater than element at middle
| 28|         # Do Binary Search on right sub array because only
| 29|         # right sub array will contain the element
| 30|         else: 
| 31|             return binSe arch(list_to_search, middle + 1, right, ele)
| 32|     else: 
| 33|         # -1 value indicates that the element is not
| 34|         # found in the array 
| 35|         return -1
| 36|
| 37| # function to remove duplicates
| 38| # @param list_to_rem_dup: sorted list from which duplicates
| 39| #                         have to be removed
| 40| def remove_duplicates(list_to_rem_dup):
| 41|     # create an empty list
| 42|     new_list = []
| 43|     # For each index from end of sorted array to 1, perform
| 44|     # binary search to check if their duplicates exit
| 45|     # If the duplicates do not exits, then append
| 46|     # the element to the new_list
| 47|     # Finally, reverse the list to maintain the order
| 48|     for i in reversed(range(1, len(list_to_rem_dup))):
| 49|         curr_ele = list_to_rem_dup[i]
| 50|          found_index = binSearch(list_to_rem_dup, 0, i-1, curr_ele)
| 51|     if found_index == -1:
| 52|        new_list.append(curr_ele)
| 53|     new_list.append(list_to_rem_dup[0])
| 54|     return reversed(new_list)
| 55|
| 56| # driver method
| 57| def main():
| 58|     # list on which operation is to be performed
| 59|     ord_list = [6, 12, 12, 13, 15, 19, 19, 21] 
| 60|     # print the sorted array in order
| 61|    p rint ("\nSorted array with duplicates") 
| 62|    for ele in range(len(ord_list)): 
| 63|        print (ord_list[ele])
| 64|    # removing duplicates from list
| 65|    without_dup_list = remove_duplicates(ord_list)
| 66|    print ("\nOrdered array without duplicates") 
| 67|    for element in without_dup_list:
| 68|        print(element)
| 69|
| 70| if __name__ == '__main__':
| 71|     main()

#====================================================================================================================================================
#       :: Dynamic Programming :: 
#       :: dynamic_programming/best_time_to_buy_and_sell_stock.py ::
#       LC-121 | Best Time to Buy and Sell Stock | https://leetcode.com/problems/best-time-to-buy-and-sell-stock/ | Easy
#====================================================================================================================================================
|  1| """
|  2| You are given an array prices where prices[i] is the price of a given stock on the ith day.
|  3| 
|  4| You want to maximize your profit by choosing a single day to buy one stock and choosing a different day in the future to sell that stock.
|  5| 
|  6| Return the maximum profit you can achieve from this transaction. If you cannot achieve any profit, return 0.
|  7|
|  8| Example 1:
|  9|
| 10|   Input: prices = [7,1,5,3,6,4]
| 11|   Output: 5
| 12|   Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
| 13|   Note that buying on day 2 and selling on day 1 is not allowed because you must buy before you sell.
| 14|
| 15| Example 2:
| 16|
| 17|   Input: prices = [7,6,4,3,1]
| 18|   Output: 0
| 19|   Explanation: In this case, no transactions are done and the max profit = 0.
| 20|
| 21| """
| 22|
| 23| #Classical dynamic programming problem. Let dp[i] be a maximum profit we can have if we sell stock at i-th moment of time.
| 24| #Then we can get, that dp[i+1] = max(dp[i] + q, q), where q = nums[i+1] - nums[i], we have two choices, either we just buy and immeditely sell and get q gain, or we use dp[i] + q to merge two transactions.
| 24| #
| 25| #Note now, that we do not really need to keep all dp array, but we can keep only last state.
| 26|
| 27| #TC: O(N) - We traverse the list containing N elements only once.
| 28| #SC: O(1) - The variables are constant during each iteration of our traversal.
| 29| import math
| 30| from typing import List
| 31|
| 32| def maxProfit(prices: List[int]) -> int:
| 33|     ans, dp = 0, 0
| 34|     for i in range(0, len(nums)-1):
| 35|         q = nums[i+1] - nums[i]
| 36|         dp = max(dp + q, q)
| 37|         ans = max(ans, dp)
| 38|     return ans

#====================================================================================================================================================
#       :: Dynamic Programming :: 
#       :: dynamic_programming/best_time_to_buy_and_sell_stock_ii.py ::
#       LC-122 | Best Time to Buy and Sell Stock II | https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii | Easy
#====================================================================================================================================================
|  1| """
|  2| You are given an array prices where prices[i] is the price of a given stock on the ith day.
|  3| 
|  4| Find the maximum profit you can achieve. You may complete as many transactions as you like (i.e., buy one and sell one share of the stock multiple times).
|  5| 
|  6| Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
|  7|
|  8| Example 1:
|  9|
| 10|   Input: prices = [7,1,5,3,6,4]
| 11|   Output: 7
| 12|   Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
| 13|   Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
| 14|
| 15| Example 2:
| 16|
| 17|   Input: prices = [1,2,3,4,5]
| 18|   Output: 4
| 19|   Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
| 20|   Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.
| 21|
| 22| Example 3:
| 23|
| 24|   Input: prices = [7,6,4,3,1]
| 25|   Output: 0
| 26|   Explanation: In this case, no transaction is done, i.e., max profit = 0.
| 27|
| 28| """
| 29|
| 30| import math
| 31| from typing import List
| 32| # Approach 1: Greedy
| 33| # -------------------------------
| 34| # Scan through the prices time series and capture the positive profit.
| 35| #
| 36| #TC: O(N) - We traverse the list containing N elements only once.
| 37| #SC: O(1) - The variables are constant during each iteration of our traversal.
| 38| def maxProfit(prices: List[int]) -> int:
| 39|     return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))
| 40|
| 41| #Alternatively,
| 42| def maxProfit(prices: List[int]) -> int:
| 43|     ans = 0
| 44|     for i in range(1, len(prices)):
| 45|         ans += max(0, prices[i] - prices[i-1])
| 46|     return ans
| 47|
| 48| # Approach 2: Dynamic Programming
| 49| # -------------------------------
| 50| # An alternative approach would be to use Dynamic Programming.
| 51| # Here, we define two state variables buy and sell which represent series of transactions ended with buy or sell of a stock.
| 52| # Here, it is a bit tricky to define sell as one cannot sell a stock without buying it first.
| 53| # In fact, it is not difficult to figure out that sell = 0. The transition between states is satisfied.
| 54| #
| 52| #TC: O(N) - We traverse the list containing N elements only once.
| 53| #SC: O(1) - The variables are constant during each iteration of our traversal.
| 54| def maxProfit(prices: List[int]) -> int:
| 55|     buy, sell = -prices[0], 0
| 56|     for i in range(1, len(prices)):
| 57|         buy, sell = max(buy, sell - prices[i]), max(sell, buy + prices[i])
| 58|     return sell

#====================================================================================================================================================
#       :: Dynamic Programming :: 
#       :: dynamic_programming/best_time_to_buy_and_sell_stock_iii.py ::
#       LC-123 | Best Time to Buy and Sell Stock III | https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii | Easy
#====================================================================================================================================================
|  1| """
|  2| You are given an array prices where prices[i] is the price of a given stock on the ith day.
|  3| 
|  4| Find the maximum profit you can achieve. You may complete at most two transactions.
|  5| 
|  6| Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
|  7|
|  8| Example 1:
|  9|
| 10|   Input: prices = [3,3,5,0,0,3,1,4]
| 11|   Output: 6
| 12|   Explanation: Buy on day 4 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
| 13|   Then buy on day 7 (price = 1) and sell on day 8 (price = 4), profit = 4-1 = 3.
| 14|
| 15| Example 2:
| 16|
| 17|   Input: prices = [1,2,3,4,5]
| 18|   Output: 4
| 19|   Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
| 20|   Note that you cannot buy on day 1, buy on day 2 and sell them later, as you are engaging multiple transactions at the same time. You must sell before buying again.
| 21|
| 22| Example 3:
| 23|
| 24|   Input: prices = [7,6,4,3,1]
| 25|   Output: 0
| 26|   Explanation: In this case, no transaction is done, i.e., max profit = 0.
| 27|
| 28| """
| 29|
| 30| """
| 31| This problem is a special case of problem 188. Best Time to Buy and Sell Stock IV. The following solution works for both problems.
| 32| The only difference is that here we have k = 2 transactions and in problem 188 we can have different k.
| 33|
| 34| My way to solve this problem is to first evaluate matrix B of differences and then what is asked is to find the 
| 35| maximum sum of two(k) contiguous subarrays. We can also make optimization 
| 36| if k > len(prices)//2: return sum(x for x in B if x > 0),
| 37| which will help us, if k is big (in this problem it is equal to 2, so you can remove this line,
| 38| but you need it for problem 188). If k is more than half of length of prices, we can just choose all positive elements,
| 39| we will have enough trancastions to do it.
| 40|
| 41| Let us create dp array with size k+1 by n-1, where dp[i][j] is the maximum gain, where already made j transactions,
| 42| that is choose j contiguous subarrays and used all elements before our equal number i.
| 43| Also, mp[i][j] = max(dp[0][j], ..., dp[i][j]). We take k+1 size, because we start with 0 transactions,
| 44| which will be filled with zeros. We take n-1 size, because original size is n, and size of differences is n-1.
| 45| Also we start with dp[0][1] = B[0], because we need to choose one contiguous subarray which ends with element B[0],
| 46| which is B[0] itself. Also we put mp[0][1] = B[0] for the same logic.
| 47|
| 48| Now, about updates: we iterate over all i from 1 to n-2 inclusive and j from 1 to k inclusive and:
| 49|
| 50| 1. Update dp[i][j] = max(mp[i-1][j-1], dp[i-1][j]) + B[i]. By definition we need to take B[i].
| 51|    We can either say, that we add it to the last contiguous subarray: dp[i-1][j] + B[i],
| 52|    or we say that it is new contiguous subarray: mp[i-1][j-1] + B[i]. Note, here we use mp,
| 53|    because we actually have max(dp[0][j-1], ... , dp[i-1][j-1]).
| 54| 2. Update mp[i][j] = max(dp[i][j], mp[i-1][j]).
| 55| 3. Finally, return maximum from the mp[-1], we need to choose maximum, because optimal solution can be with less than k transactions.
| 56|
| 57| Complexity: Time complexity is O(nk) = O(n), because here k = 2. Space complexity is also O(nk) = O(n).
| 58| """
| 59|
| 60| #TC: O(N*K) = O(N), because here k = 2
| 61| #SC: O(N*K) = O(N)
| 62| from typing import List
| 63| def maxProfit(prices: List[int]) -> int:
| 64|     if len(prices) <= 1: return 0
| 65|     n, k = len(prices), 2
| 66|
| 67|     B = [prices[i+1] - prices[i] for i in range(len(prices) - 1)]
| 68|     if k > len(prices)//2: return sum(x for x in B if x > 0)
| 69|    
| 70|     dp = [[0]*(k+1) for _ in range(n-1)] 
| 71|     mp = [[0]*(k+1) for _ in range(n-1)] 
| 72|
| 73|     dp[0][1], mp[0][1] = B[0], B[0]
| 74|
| 75|     for i in range(1, n-1):
| 76|         for j in range(1, k+1):
| 77|             dp[i][j] = max(mp[i-1][j-1], dp[i-1][j]) + B[i]
| 78|             mp[i][j] = max(dp[i][j], mp[i-1][j])
| 79|
| 80|     return max(mp[-1])
| 81|
| 82| """
| 83| Optimization:
| 84|
| 85| Note, that if we have array like [1,5,7, -7, -4, -3, 10, 2, 7, -4, -8, 13, 15], then we can 
| 86| work in fact with smaller array [1+5+7, -7-4-3, 10+2+7, -4-8, 13+15] = [13,-14,19,-12,28].
| 87| So, instead of :
| 88|
| 89| B = [prices[i+1] - prices[i] for i in range(len(prices) - 1)]
| 90|
| 91| we can evaluate:
| 92|
| 93| delta = [prices[i+1]-prices[i] for i in range (len(prices)-1)]
| 94| B=[sum(delta) for _, delta in groupby(delta, key=lambda x: x < 0)]
| 95| n, k = len(B) + 1, 2
| 96| """
| 97| # Optimized Solution ( working on smaller array )
| 98| from typing import List
| 99| def maxProfit(prices: List[int]) -> int:
|100|     if len(prices) <= 1: return 0
|101|     n, k = len(prices), 2
|102|
|103|     delta = [prices[i+1]-prices[i] for i in range (len(prices)-1)]
|104|     B = [sum(delta) for _, delta in groupby(delta, key=lambda x: x < 0)]
|105|     n, k = len(B) + 1, 2
|106|     if k > len(prices)//2: return sum(x for x in B if x > 0)
|107|    
|108|     dp = [[0]*(k+1) for _ in range(n-1)] 
|109|     mp = [[0]*(k+1) for _ in range(n-1)] 
|110|
|111|     dp[0][1], mp[0][1] = B[0], B[0]
|112|
|113|     for i in range(1, n-1):
|114|         for j in range(1, k+1):
|115|             dp[i][j] = max(mp[i-1][j-1], dp[i-1][j]) + B[i]
|116|             mp[i][j] = max(dp[i][j], mp[i-1][j])
|117|
|118|     return max(mp[-1])
        
#====================================================================================================================================================
#       :: Dynamic Programming :: 
#       :: dynamic_programming/best_time_to_buy_and_sell_stock_iv.py ::
#       LC-188 | Best Time to Buy and Sell Stock IV | https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv | Easy
#====================================================================================================================================================
|  1| """
|  2| You are given an integer array prices where prices[i] is the price of a given stock on the ith day, and an integer k.
|  3| 
|  4| Find the maximum profit you can achieve. You may complete at most k transactions.
|  5| 
|  6| Note: You may not engage in multiple transactions simultaneously (i.e., you must sell the stock before you buy again).
|  7|
|  8| Example 1:
|  9|
| 10|   Input: k = 2, prices = [2,4,1]
| 11|   Output: 2
| 12|   Explanation: Buy on day 1 (price = 2) and sell on day 2 (price = 4), profit = 4-2 = 2.
| 13|
| 14| Example 2:
| 15|
| 16|   Input: k = 2, prices = [3,2,6,5,0,3]
| 17|   Output: 7
| 18|   Explanation: Buy on day 2 (price = 2) and sell on day 3 (price = 6), profit = 6-2 = 4. Then buy on day 5 (price = 0) and sell on day 6 (price = 3), profit = 3-0 = 3.
| 19|
| 28| """
| 29|
| 30| #First Approach: Buy Low & Sell High
| 32| #For the first approach:
| 33| #NOTE: Approach could be generalized into cases with multiple transactions, say K. ( So, TC: O(N*K) and O(N), where, K = number of transactions. ) 
| 34| #TC: O(N*K)
| 35| #SC: O(K)
| 36| import math
| 37| from typing import List
| 38| def maxProfit(k: int, prices: List[int]) -> int:
| 35|     if k >= len(prices)//2: 
| 36|         return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))
| 37|
| 38|     buy, pnl = [inf]*k, [0]*k
| 39|     for price in prices:
| 40|         for i in range(k):
| 41|             buy[i] = min(buy[i], price - (pnl[i-1] if i else 0))
| 42|             pnl[i] = max(pnl[i], price - buy[i])
| 43|         return pnl[-1] if prices and k else 0
| 44|
| 45| #Second Approach Kadane's Algorithm
| 46| #The second approach is to extend Kadane's algorithm to the cases with multiple transaction (O(N*K) time & O(N) space).
| 47| #TC: O(N*K)
| 48| #SC: O(N)
| 49| import math
| 50| from typing import List
| 51| def maxProfit(k: int, prices: List[int]) -> int:
| 52|     if k >= len(prices)//2: 
| 53|         return sum(max(0, prices[i] - prices[i-1]) for i in range(1, len(prices)))
| 54|
| 55|     pnl = [0]*len(prices)
| 56|     for _ in range(k):
| 57|         val = 0
| 58|         for i in range(1, len(prices)):
| 59|             val = max(pnl[i], val + prices[i] - prices[i-1])
| 60|             pnl[i] = max(pnl[i-1], val)
| 61|     return pnl[-1]

#===========================================================================================================================================================
#       :: Dynamic Programming :: 
#       :: dynamic_programming/best_time_to_buy_and_sell_stock_with_cooldown.py ::
#       LC-309 | Best Time to Buy and Sell Stock With Cooldown | https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown | Easy
#===========================================================================================================================================================
|  1| """
|  2| For all this buy and sell stocks problems I prefer to use differences array.
|  3| For example, if you have [1,2,3,1,4], then we have [1, 1, -2, 3] differences. Then the goal is to take as many of subarrays (with adjacent elements) 
|  4| with biggest sum, such that there is not gap with size 1. For example, for given differences, we can not take [1,1] and [3], 
|  5| but we can take [1] and [3], so the answer will be 4.
|  6| 
|  7| Let n be number of elements in prices, than there will be n-1 elements in diff array. 
|  8| Let us create dp and dp_max arrays with n+1 elements, that is two extra elements, such that
|  9| 
| 10| dp[i] is maximum gain for first i elements of diff, where we use i-th element
| 11| dp_max[i] is maximum gain for first i elements of diff (we can use i and we can not use it).
| 12|
| 13| Now, we can do the following steps:
| 14| 
| 15| dp[i] = diff[i] + max(dp_max[i-3], dp[i-1]), because, first of all we need to use i,
| 16| so we take diff[i]. Now we have two options: skip 2 elements and take dp_max[i-3], or do not skip anything and take dp[i-1].
| 17| Update dp_max[i] = max(dp_max[i-1], dp[i]), standard way to update maximum.
| 18| Finally, we added 2 extra elements to dp and dp_max, so instead of dp_max[-1] we need to return dp_max[-3].
| 19|
| 20| Complexity: both time and space complexity is O(n). Space complexity can be improved to O(1), because we look back at only 3 elements.
| 21| """
| 22|
| 23| # DP Approach
| 24| # TC: O(N)
| 25| # SC: O(1) - because we look back at only 3 elements
| 26| from typing import List
| 27| def maxProfit(prices: List[int]) -> int:
| 28|     n = len(prices)
| 29|     if n <= 1: return 0
| 30|   
| 31|     diff = [prices[i+1] - prices[i] for i in range(n-1)]
| 32|     dp, dp_max = [0]*(n + 1), [0]*(n + 1)
| 33|     for i in range(n-1):
| 34|         dp[i] = diff[i] + max(dp_max[i-3], dp[i-1])
| 35|         dp_max[i] = max(dp_max[i-1], dp[i])
| 36| 
| 37|     return dp_max[-3]
    
#===========================================================================================================================================================
#       :: Dynamic Programming :: 
#       :: dynamic_programming/best_time_to_buy_and_sell_stock_with_transaction_fee.py ::
#       LC-714 | Best Time to Buy and Sell Stock With Transaction Fee | https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/ | Easy
#===========================================================================================================================================================
|  1| """
|  2| Let us define diff[i] = prices[i+1] - prices[i]. Then we need to find maximum gain for several continuous subarrays, where we pay fee for each subarray. Let us consider it on example:
|  3| [1, 3, 5, 4, 8, 7, 4], then differences will be [2, 2, -1, 4, -1, -3]. For example we can take two subarrays [2, 2] and [4], it means we make two trancastions and we need to pay 2 fees. In original array it means we buy at price 1, then sell at price 5. Then buy at price 4 and sell at price 8.
|  4|
|  5| Use dynamic programming, where dp[i] is maximum gain at i-th moment of time if we use diff[i] and sp[i] be maximum among dp[0], ..., dp[i], that is running maximum, that is sp[i] is the gain we can get, using first i times.
|  6| Then we can have 2 options:
|  7|
|  8|   1. We continue last subarray, so we get diff[i] + dp[i-1].
|  9|   2. We start new subarray, so we get diff[i] + sp[i-2] - fee: here we take sp[i-2], because we need to skip one element, so subarrays are separated.
| 10|
| 11| Let us look at our example with differences [2, 2, -1, 4, -1, -3]:
| 12| 
| 13| 1. dp[0] is the maximum gain we can get using ony first difference, we can have 2 and we need to pay fee, so we have 1.
| 14| 2. dp[1] is maxumum gain we can get using [2, 2]. We can continue previous transaction, so we will gain 3. Or we can try to start new transaction. However it is not possible, because we need to have a gap here.
| 15| 3. dp[2] is maximum gain we can get using [2, 2, -1]. Note, that by definition of dp[i] we need to use last element here, so again we have two choices: if we continue first transaction, we have 3-1 = 2 gain. Or we start new transaction, and then we need to make gap and previous transaction will be for element i-2 or smaller. Exaclty for this we use sp: running maximum of dp. In our case sp[0] = 1, so total gain if we start new transaction is sp[0] - fee + -1 = -1.
| 16| 4. dp[3] is maximum gain we can get using [2, 2, -1, 4]. Again, we can have two choices: continue last transaction, in this case we have dp[2] + 4 = 6. If we start new transaction, we have sp[1] - fee + 4 = 6 as well. So in this case does not matter, what option we choose and it makes sence: fee is equal to decrease of price.
| 17| 5. dp[4] is maximum gain we can get using [2, 2, -1, 4, -1]. Again, we have choice between dp[3] + -1 = 5 and sp[2] - fee + -1 = 4.
| 18| 6. dp[5] is maximum gain we can get using [2, 2, -1, 4, -1, -3]. We have either dp[4] + -3 = 2 or sp[3] - fee + -3 = 2.
| 19|
| 20| Finally, we have arrays like this:
| 21|
| 22| dp = [1, 3, 2, 6, 5, 2, -inf]
| 22| sp = [1, 3, 3, 6, 6, 6, 0]
| 23|
| 24| Complexity: time and space complexity is O(n), where space complexity can be reduced to O(1), because we look back at only 2 elements.
| 25| """
| 26|
| 27| # DP Solution
| 28| # TC: O(N)
| 29| # TC: O(1) - because we look back at only 2 elements
| 30| def maxProfit(prices: List[int], fee: int) -> int:
| 31|     if len(prices) == 1: return 0
| 32|     n = len(prices)
| 33|
| 34|     dp, sp = [-float(inf)]*n, [0]*n
| 35|
| 37|     for i in range(n-1):
| 38|         dp[i] = prices[i+1] - prices[i] + max(dp[i-1], sp[i-2] - fee)
| 39|         sp[i] = max(sp[i-1], dp[i])
| 40|
| 41|     return sp[-2] 

