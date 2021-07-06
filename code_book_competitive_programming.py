# --- Code Book: Competitive Programming
# --- Source   : github.com/sm2774us/competitive_programming_book.git

#====================================================================================================================================================
#       :: Arrays ::
#       :: Kadane's Algorithm :: 
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
#       :: Kadane's Algorithm ::
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
| 15| # TC: O(N)
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
#       :: Prefix Sum + Modular Arithmetic ::
#       :: array/continuous_subarray_sum.py ::
#       LC-523 | Continuous Subarray Sum | https://leetcode.com/problems/continuous-subarray-sum/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given an integer array nums and an integer k, return true if nums has a continuous subarray of size at least two whose elements 
|  3| sum up to a multiple of k, or false otherwise.
|  4|
|  5| An integer x is a multiple of k if there exists an integer n such that x = n * k. 0 is always a multiple of k.
|  6|
|  7| Example 1:
|  8|
|  9|   Input: nums = [23,2,4,6,7], k = 6
| 10|   Output: true
| 11|   Explanation: [2, 4] is a continuous subarray of size 2 whose elements sum up to 6.
| 12|
| 13| Example 2:
| 14|
| 15|   Input: nums = [23,2,6,4,7], k = 6
| 16|   Output: true
| 17|   Explanation: [23, 2, 6, 4, 7] is an continuous subarray of size 5 whose elements sum up to 42.
| 18|   42 is a multiple of 6 because 42 = 7 * 6 and 7 is an integer.
| 19|
| 20| Example 3:
| 21|
| 22|   Input: nums = [23,2,6,4,7], k = 13
| 23|   Output: false
| 24| """
| 25|
| 26| """
| 27| Prefix Sum Review
| 28| ---------------------------------
| 29| We can solve this question with a prefix sum, an array where each position is the accumulated sum until the current index, but not including it.
| 30| 
| 31| For example:
| 32|
| 33|   * arr = [23, 2, 4, 6, 7]
| 34|   * prefix_sum = [0, 23, 25, 29, 35, 42]
| 35|
| 36| And, with prefix sum, we can also query the contiguous subarray sum between two index. Following the same example:
| 37|
| 38|   prefix_sum = [0, 23, 25, 29, 35, 42]
| 39|   # Sum between element 1 and 3: (2, 4, 6)
| 40|   prefix_sum[3 + 1] - prefix_sum[1] = 35 - 23 = 12
| 41|
| 42|   * prefix_sum[3 + 1] because we want to include the element at index 3, so : (23 + 2 + 4 + 6 + 7).
| 43|   * prefix_sum[1] because we want to remove all elements until the element at idx 1: (23)
| 44|   * prefix_sum[4] - prefix_sum[1] = (23 + 2 + 4 + 6 + 7) - 23 = 2 + 4 + 6 + 7
| 45|
| 46| Question's Idea
| 47| ---------------------------------
| 48| Now we know more about the prefix_sum, lets think what is have a sum as multiple of k. Given the example nums = [23, 2, 4, 6, 7], k = 6, all valid sums are:
| 49|
| 51|   * (2 + 4) = 6
| 52|   * (6) = 6
| 53|   * (23 + 2 + 4 + 6 + 7) = 42
| 54|
| 55|   We also can use a modulo property, where: 
| 56|
| 57|   (A + B) % K = (A % K + B % K) % K
| 58|
| 59|   So we can use a modulo in our accumulated array and, instead of looking for a sum equals to k*n, we want to find a sum equals to k.
| 60|
| 61| Summarizing everything:
| 62| ---------------------------------
| 64| arr = [23, 2, 4, 6, 7]
| 65| prefix_sum = [0, 23, 25, 29, 35, 42]
| 66| modulo_prefix_sum = [0, 5, 1, 5, 5, 0]
| 67|
| 68| But wait! Now we don't have a valid prefix_sum, since it isn't crescent.
| 69| Yes, and that's why we want to look forward similar values. For example, we have two 5's. That means that, in my prefix_sum, 
| 70| I had two values where sum % k = 5. But what are we looking for now? Let's think:
| 71|
| 72|  A + B = n * k # simbolizes that A + B are multiple of k (k * n is equal to that sum)
| 73|  (A + B) % k = (n * k) % k
| 74|  (A % k + B % k) % k = n % k * k % k
| 75|  (A % k + B % k) % k = 0 # (OMG!!)
| 76|
| 77| Using the previous idea, we know now that we have to find a subarray where (nums[i] % k + nums[i+1] % k + ... nums[j] % k) % k == 0.
| 78| But, knowing our prefix_sum, we know that prefix_sum[j + 1] - prefix_sum[i] is exactly that sum!!!.
| 79| So, using everything we already learned, prefix_sum[j + 1] == prefix_sum[i]!!!!!!!!!!
| 80|
| 81| Conclusion:
| 82| ---------------------------------
| 83| So, thanks to the explanation before, we are looking for equal values in our custom prefix_sum (prefix_sum with modulos) 
| 84| and i and j cannot be consecutives (if j = i + 1, prefix_sum[i + 1] - prefix_sum[i] means the sum between [i, i], a unique value)
| 85| """
| 86|
| 87| #Prefix Sum + Modular Arithmetic Solution
| 88| #TC: O(N)
| 89| #SC: O(N)
| 90| from typing import List
| 91| def checkSubarraySum(nums: List[int], k: int) -> bool:
| 92|     # We don't want to do (num % 0,) right? hehe
| 93|     if not k:
| 94|         # If k is 0 and there is no negative value, the only possible solution is a subarray like [0,0] (e.g. [1,7,3,0,0,1,2,5,0,1])
| 95|         return any([nums[i] + nums[i - 1] == 0 for i in range(1, len(nums))])
| 96|        
| 97|     prefix_sum = [0 for _ in range(len(nums) + 1)]
| 98|
| 99|     for idx in range(len(nums)):
|100|         prefix_sum[idx + 1] = (prefix_sum[idx] + nums[idx] % k) % k
|101|            
|102|     memo = {}
|103|
|104|     for idx, curr_sum in enumerate(prefix_sum):
|105|         if curr_sum in memo and idx - memo[curr_sum] > 1:
|106|             return True
|107|
|108|         if curr_sum not in memo:
|109|             memo[curr_sum] = idx
|110|
|101|     return False
|102|
|103|
|104| # More concise soluton using python builtin itertools.accumulate ( Good to know but don't use in an interview setting )
|105| import itertools
|106| from typing import List
|107| def checkSubarraySum(nums: List[int], k: int) -> bool:
|108|     mapping = {0: -1}
|109|     for i, prefix_sum in enumerate(itertools.accumulate(nums)):
|110|         key = prefix_sum % k if k else prefix_sum
|111|         if key not in mapping:
|112|             mapping[key] = i
|113|         elif i - mapping[key] > 1:
|114|             return True
|115|     return False

#========================================================================================================================
#       :: Arrays ::
#       :: Partial Sum & Hash-Map ::
#       :: array/contiguous_array.py ::
#       LC-525 | Contiguous Array | https://leetcode.com/problems/contiguous-array/ | Medium
#========================================================================================================================
|  1| """
|  2| Given a binary array nums, return the maximum length of a contiguous subarray with an equal number of 0 and 1.
|  3|
|  4| 
|  5| Example 1:
|  6|
|  7|   Input: nums = [0,1]
|  8|   Output: 2
|  9|   Explanation: [0, 1] is the longest contiguous subarray with an equal number of 0 and 1.
| 10|
| 11| Example 2:
| 12|
| 13|   Input: nums = [0,1,0]
| 13|   Output: 2
| 13|   Explanation: [0, 1] (or [1, 0]) is a longest contiguous subarray with equal number of 0 and 1.
| 14|
| 12| """
| 13|
| 14| """
| 15| Hint:
| 16| ---------------------------------
| 17| Maintain a partial sum to help us to judge whether we have subarray with equal number of 0 and 1
| 18|
| 19| Initial partial sum value is 0, index is -1
| 20|
| 21| Scan each number in nums.
| 22|
| 23| If number is 1, add 1 to partial sum. ( + 1 )
| 24| If number is 0, subtract 1 from partial sum. ( - 1 )
| 25|
| 26| Once there are two indices, says i and j, with i =/= j, have the same partial sum, then we have one contiguous subarray with equal number of 0 and 1.
| 27|
| 28| Abstract Model and Diagram:
| 29| ---------------------------------
| 30|                                         +-----------------------------------------------+
| 31|                                         |             Contiguous Subarray               |
| 31|                 +-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 32|      index      |    ...    |     i     |    ...    |    ...    |    ...    |     j     |    ...    |
| 33|                 +-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 34|  Partial sum    |    ...    |     S     |    ...    |    ...    |    ...    |     S     |    ...    |
| 35|                 +-----------+-----------+-----------+-----------+-----------+-----------+-----------+
| 36|                                         |                                               |
| 37|                                         +-----------+-----------+-----------+-----------+
| 38|
| 39|                                         If partial_sum[i] = partial_sum[j]
| 40|                                         Then, there are equal number of 1s and 0s in interval from i+1 to j.
| 41|
| 42|                                         Length of subarray = j-i
| 43| """
| 44|
| 45| # Partial Sum and Hash-Map.
| 46| # TC: O(N)
| 47| # SC: O(N)
| 48| from typing import List
| 49| def findMaxLength(nums: List[int]) -> int:
| 50|     partial_sum = 0
| 51|     # table is a dictionary
| 52|     # key : partial sum value
| 53|     # value : the left-most index who has the partial sum value
| 54|     table = { 0: -1}
| 55|     max_length = 0
| 56|        
| 57|     for idx, number in enumerate( nums ):
| 58|         # partial_sum add 1 for 1
| 59|         # partial_sum minus 1 for 0
| 60|         if number:
| 61|             partial_sum += 1
| 62|         else:
| 63|             partial_sum -= 1
| 64|       
| 65|         if partial_sum in table:                
| 66|             # we have a subarray with equal number of 0 and 1
| 67|             # update max length                
| 68|             max_length = max( max_length, ( idx - table[partial_sum] ) )
| 69|         else:
| 70|             # update the left-most index for specified partial sum value
| 71|             table[ partial_sum ] = idx
| 72|
| 73|     return max_length


#========================================================================================================================
#       :: Arrays ::
#       :: Prefix Sum ::
#       :: array/maximum_size_subarray_sum_equals_k.py ::
#       LC-325 | Maximum Size Subarray Sum Equals k | https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/ | Medium
#========================================================================================================================
|  1| """
|  2| Given an array nums and a target value k, find the maximum length of a subarray that sums to k.
|  3| If there isn't one, return 0 instead.
|  4|
|  5| Note:
|  6| The sum of the entire nums array is guaranteed to fit within the 32-bit signed integer range.
|  7|
|  8| Example 1:
|  9| Given nums = [1, -1, 5, -2, 3], k = 3, return 4. (because the subarray [1, -1, 5, -2] sums to 3 and is the longest)
| 10| """
| 11|
| 12| #TC: O(N)
| 13| #SC: O(N)
| 14| from typing import List
| 15| def maxSubArrayLen(self, nums: List[int], k: int) -> int:
| 16|     sums = {}
| 17|     cur_sum, max_len = 0, 0
| 18|     for i in range(len(nums)):
| 19|         cur_sum += nums[i]
| 20|         if cur_sum == k:
| 21|             max_len = i + 1
| 22|         elif cur_sum - k in sums:
| 23|             max_len = max(max_len, i - sums[cur_sum - k])
| 24|         if cur_sum not in sums:
| 25|             sums[cur_sum] = i  # Only keep the smallest index.
| 26|     return max_len

#====================================================================================================================================================
#       :: Arrays ::
#       :: Prefix Sum :: 
#       :: array/subarray_sum_divisible_by_k.py ::
#       LC-974 | Subarray Sum Divisible by K | https://leetcode.com/problems/subarray-sums-divisible-by-k/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given an array nums of integers, return the number of (contiguous, non-empty) subarrays that have a sum divisible by k.
|  3|
|  4| Example 1:
|  5|
|  6|   Input: nums = [4,5,0,-2,-3,1], k = 5
|  7|   Output: 7
|  8|   Explanation: 
|  9|     There are 7 subarrays with a sum divisible by k = 5:
| 10|     [4, 5, 0, -2, -3, 1], [5], [5, 0], [5, 0, -2, -3], [0], [0, -2, -3], [-2, -3]
| 11|
| 12| Note:
| 13|   * 1 <= nums.length <= 30000
| 14|   * -10000 <= nums[i] <= 10000
| 15|   * 2 <= k <= 10000
| 16| """
| 17|
| 18| """
| 19| Solution Explanation:
| 20| ------------------------------------------------
| 21| Subarray sums divisible by k:
| 22| ------------------------------------------------
| 23|   arr = [ a_0, a_1, ... , a_i, ... , a_j, ... , a_n ], k
| 24|   we want all subarrays
| 25|   a[i:j] such that sum(a[i:j])%k = 0 
| 26|
| 27| Method:
| 28| ------------------------------------------------
| 29|   * Calculate prefix sum for the array:
| 30|      F = [ 0, a_1, a_1+a_2, a_1+a_2+a_3, ... ]
| 31|     so, if we have F[i], then
| 32|     we need to find an F[j],
| 33|     where, j < i, such that:
| 34|
| 35|       1. ( F[i] - F[j] ) % k = 0
| 36|                     |
| 37|                     ▼
| 38|       2.     F[i] % k  =  F[j] % k
| 39|              (___ ___)    (___ ___)
| 40|                  ▼            ▼
| 41|            Calculate       Store in
| 42|         as you traverse    hash-map
| 43|         through array
| 44|
| 45|   How did we go from (1) to (2) ? 
| 46|
| 47|   ➜ ( F[i] - F[j] ) % k = 0                       \    Example:
| 48|            means                                   |    (10-4)%2 = 0
| 49|        F[i] - F[j] = k*n, where n is some number    >   means
| 50|                |                                   |    10 - 4 = 2 * x
| 51|                ▼                                   /         6 = 2 * x
| 52|         F[j] = F[i] - k*n                                    x = 3
| 53|                | mod both sides by k
| 54|                ▼
| 55|       F[j] % k = ( F[i] - k*n ) % k
| 56|                  (________ ________) 
| 57|                           ▼
| 58|                   apply distibutive
| 59|                   law of mod
| 60|                     (a + b) % k = (a%k + b%k) % k
| 61|                 |                 (_______ ______)
| 62|                 |                         ▼
| 63|                 ▼
| 64|         F[j] % k = ( F[i] % k ) % k
| 65|                    (_______ _______)
| 66|                            ▼
| 67|                     apply another law
| 68|                     of mods: (a % b) % b = a % b
| 69|                 |                         ▼
| 70|                 ▼
| 71|       +---------------------+
| 72|       | F[j] % k = F[i] % k |
| 73|       +---------------------+
| 74|
| 75|  ➜ Algorithm:             Why ? Example: [3,6] w/ k=3
| 76| ------------------------------------------------
| 77|     ➜ prefix_sum = 0   🢇              if we don't instantiate
| 78|     ➜ sums = {0:1}   🢇                sums to {0,1}, then
| 79|     ➜ answer = 0                      our answer = 1 ≠ correct = 3
| 80|                                                        answer
| 81|     for i in range(len(arr)):
| 82|         prefix_sum ± arr[i]
| 83|         key = prefix_sum % k <= F[i] % k
| 84|         if key in sums:       <= F[j] % k
| 85|             answer ± sums[key]  <===== there could
| 86|         sums[key] ± 1            be more than one
| 87|     return answer                F[j] where F[j] % k = F[i] % k
| 88|
| 89| Also, to clarify, (F[i] - F[j]) where j < i , is equal to the sum of the subarray from arr[j] to arr[i] , i.e.,
| 90| sum(arr[j:i]) , not inclusive of i.
| 91|
| 92| Example:
| 93| ------------------------------------------------
| 94| arr = [1, 2, 3, 4]
| 95| F = [0, 1, 3, 6, 10]
| 96| so, for example, F[3]-F[1] = 6-1 = 5 = sum(arr[1:3])
| 97| """
| 98|
| 99| # Prefix Sum Approach
|100| # TC: O(N)
|101| # SC: O(N)
|102| def subarraysDivByK(A: List[int], K: int) -> int:
|103|     prefix_sum = 0
|104|     sums = {0: 1}
|105|     answer = 0
|106|     for num in A:
|107|         prefix_sum += num
|108|         key = prefix_sum%K
|109|         if key in sums:
|110|             answer += sums[key]
|111|             sums[key] += 1
|112|             continue
|113|         sums[key] = 1
|114|     return answer

#====================================================================================================================================================
#       :: Arrays :: 
#       :: Prefix Sum ::
#       :: array/make_sum_divisible_by_p.py ::
#       LC-1590 | Make Sum Divisible by P | https://leetcode.com/problems/make-sum-divisible-by-p/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given an array of positive integers nums, remove the smallest subarray (possibly empty) such that the sum of the remaining elements is divisible by p.
|  3| It is not allowed to remove the whole array.
|  4|
|  5| Return the length of the smallest subarray that you need to remove, or -1 if it's impossible.
|  6|
|  7| A subarray is defined as a contiguous block of elements in the array.
|  8|
|  9| Example 1:
| 10|
| 11|   Input: nums = [3,1,4,2], p = 6
| 12|   Output: 1
| 13|   Explanation: The sum of the elements in nums is 10, which is not divisible by 6. We can remove the subarray [4],
| 14|   and the sum of the remaining elements is 6, which is divisible by 6.
| 15|
| 16| Example 2:
| 17|
| 18|   Input: nums = [6,3,5,2], p = 9
| 19|   Output: 2
| 20|   Explanation: We cannot remove a single element to get a sum divisible by 9. The best way is to remove the subarray [5,2], leaving us with [6,3] with sum 9.
| 21|
| 22| Example 3:
| 23|
| 24|   Input: nums = [1,2,3], p = 3
| 25|   Output: 0
| 26|   Explanation: Here the sum is 6. which is already divisible by 3. Thus we do not need to remove anything.
| 27|
| 28| Example 4:
| 29|
| 30|   Input: nums = [1,2,3], p = 7
| 31|   Output: -1
| 32|   Explanation: There is no way to remove a subarray in order to get a sum divisible by 7.
| 33|
| 34| Example 5:
| 35|
| 36|   Input: nums = [1000000000,1000000000,1000000000], p = 3
| 37|   Output: 0
| 38| 
| 39| Constraints:
| 40|
| 41|   * 1 <= nums.length <= 105
| 42|   * 1 <= nums[i] <= 109
| 43|   * 1 <= p <= 109
| 44| """
| 45|
| 46| """
| 47| Make sum divisible by p:
| 48|   * nums = [ a_1, a_2, ..., a_j, ..., ..., a_(n-1) ], k
| 49|   * p
| 50| Step 1)  Get r, where r = sums(nums) % p
| 51|      |
| 52|      +-> if r == 0, return 0
| 53|
| 54|      |
| 55|      +-> Otherwise, we want the
| 56|          smallest subarray, nums[j:i],
| 57|          that:
| 58|          * ( sum(nums) - sum(nums[j:i]) ) % p = 0
| 59|                         (______ ______) 
| 60|                                ▼
| 61|                         can be represented
| 62|                         in terms of prefix
| 63|                         sums as: F[i] - F[j]
| 64|                           ex: * nums = [1,2,3]
| 65|                               * F = [0,1,3,6]
| 66|                               * suns(nums[0:2]) = F[2] - F[0] = 3 = 1+2
| 67|
| 68|          * ( sum(nums) - (F[i]-F[j]) ) % p = 0
| 69|                             |
| 70|                             ▼
| 71|          * sum(nums) - (F[i]-F[j]) =  n  *  p
| 72|                                      \__/ 
| 73|                                     some integer
| 74|                 | rearrange
| 75|                 | the equation
| 76|                 ▼
| 77|          * F[i] - F[j] = sum(nums) - n*p
| 78|
| 79|                     | mod both
| 80|                     | sides by p
| 81|                     ▼
| 82|          * (F[i] - F[j]) % p = ( sum(nums) - n*p ) % p
| 83|                                (___________ __________)
| 84|                                            ▼
| 85|                                 apply distibutive law
| 86|                                 of mods: (a + b) % c = (a%c + b%c) % c
| 87|                          |
| 88|                          ▼
| 79|          * (F[i] - F[j]) % p = ( sum(nums)%p  -  n*p % p ) % p
| 80|                                                 (___ ___)
| 81|                                                     ▼
| 82|                                                 cancels out,
| 83|                                                 e.g., n=3, p=2 : 3 * 2 = 6 % 2 = 0
| 84|                          |
| 85|                          ▼
| 83|          * (F[i] - F[j]) % p = ( sum(nums)%p ) % p
| 84|                                (_________ ________)
| 85|                                          ▼
| 86|                                  another mod law:
| 87|                                    (a%b)%b = a%b
| 88|                          |
| 89|                          ▼
| 90|          * (F[i] - F[j]) % p = sum(nums) % p
| 91|                               (______ ______)
| 92|                                      ▼
| 93|                                   equals r
| 94|                          |
| 95|                          ▼
| 96|          * F[i] - F[j] = n * p + r
| 97|
| 98|                          | rearrange
| 99|                          ▼
|100|          * F[j] = F[i] - n*p - r
|101|
|102|                            mod both
|103|                          | sides by p again
|104|                          ▼
|105|                                 +---------+
|105|          * F[j] % p = (F[i]%p - |(n*p % p)| - r
|106|            (___ ___)            +---------+
|107|                ▼
|108|          store in   = (F[i]%p - sum(nums)%p%p) % p
|109|        a hashmap/array
|110|
|111|                     = (F[i]%p - sum(nums)%p) % p
|112|                       (____________ ____________)
|113|                                    ▼
|114|                          key that will be
|115|                          calculated for every
|116|                          element in nums
|117| """
|118|
|119| #Prefix Sum Approach
|120| #TC: O(N)
|121| #SC: O(N)
|122| def minSubarray(self, nums: List[int], p: int) -> int:
|123|     n = len(nums)
|124|     target = sum(nums)%p
|125|     if not target:
|126|         return 0
|127|     answer = n
|128|     prefix_sum = 0
|129|     hashmap = {0: -1}
|130|     for i, num in enumerate(nums):
|131|         prefix_sum += num
|132|         key = (prefix_sum%p - target)%p
|133|         if key in hashmap:
|134|             answer = min(answer, i-hashmap[key])
|135|         hashmap[prefix_sum%p] = i
|136|     return answer if answer < n else -1

#====================================================================================================================================================
#       :: Arrays ::
#       :: Prefix Sum ::
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
| 10| """
| 11| Solution Explanation:
| 12| ------------------------------------------------
| 13| Subarray sums = K:
| 14|   Given an array of integers, nums
| 15|   We want all subarrays of nums such that sum(nums[j:i]) == k,
| 16|   ( where 0 <= j <= i < N ... where, N = length of nums )
| 17|
| 18| Method : 
| 19|   Generate a prefix_sum_array:
| 20|     * nums = [a_i, a_2, ... , a_j, ... , a_i, a_n ]
| 21|     * prefix_sum_array = F = [ 0, a_1, a_1+a+2, a_1+a_2+a_3, ... ]
| 22|
| 23| Why? :
| 24|   With F, you can express sum(nums[i:j]) as (F[i] - F[j]),
| 25|   and now, we are looking for all F[i] - F[j] , where j < i , such that:
| 26|     * F[i] - F[j] = k
| 27|     * F[j] = F[i] - k
| 28|       where, F[j] => stored in hashmap
| 29|              F[i] => calculated as you traverse the array
| 30|
| 31| So, if there is a (previous_prefix_sum - sum) (or sums, as there can be more than one sinces nums can have negative numbers)
| 32| that equals the ((current_prefix_sum - sum) - k), then the subarray from nums[j:i] sums to k.
| 33|
| 34| There are number of other prefix-sum questions, and the trick is to generate an equation (based on the problem statement), and then
| 35| rearrange the equation so that F[j] is on one side, and F[i] is on the other. Whatever is on the F[j] side are the "keys" that you
| 36| will store in a hash-map, and whatever is on the F[i] side is what you will calculate at every step of the array traversal.
| 37|
| 38| Similar problems using the prefix-sum technique:
| 39|   * LC-974 - Subarray sums divisible by k       : https://leetcode.com/problems/subarray-sums-divisible-by-k/
| 40|   * LC-1590 - Make sum divisible by p           : https://leetcode.com/problems/make-sum-divisible-by-p/
| 41|   * LC-523 - Continuous Subarray Sum            : https://leetcode.com/problems/continuous-subarray-sum/
| 42|   * LC-525 - Contiguous Array                   : https://leetcode.com/problems/contiguous-array/
| 43|   * LC-325 - Maximum Size Subarray Sum Equals k : https://leetcode.com/problems/maximum-size-subarray-sum-equals-k/
| 44| """
| 45| #Logic prefix sum and HashMap ( i.e., dictionary )
| 46| #
| 47| #TC: O(N)
| 48| #SC: O(N)
| 49| import collections
| 50| from typing import List
| 51| def subarraySum(nums: List[int], k: int) -> int:
| 52|     n = len(nums)
| 53|     prefix_sum _array = collections.defaultdict(int)
| 54|     prefix_sum_array[0] = 1
| 55|     prefix_sum = 0
| 56|     res = 0
| 57|     for i in range(n):
| 58|         prefix_sum += nums[i]
| 59|         key = prefix_sum-k
| 60|         if prefix_sum_array[key]:
| 61|             res += prefix_sum_array[key]
| 62|         prefix_sum_array[prefix_sum] += 1
| 63|     return res

#====================================================================================================================================================
#       :: Arrays ::
#       :: Sliding Window ::
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
#       :: Prefix Sum and Hash-Map ::
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
| 14| #Logic prefix sum and HashMap ( i.e., dictionary )
| 15| #Algorithm Steps:
| 16| #1. Iterate over the list.
| 17| #   1.1. Copy values of prefix_sum into a result array and reset prefix_sum,
| 18| #        each time we encounter a match ( curr_sum - k ). 
| 19| #2. The result array will have the non-overlapping subarrays.
| 20| #3. Sum up the values in the result array and return it. 
| 21| #
| 22| #TC: O(N)
| 23| #SC: O(N)
| 24| import collections
| 25| from typing import List
| 26| def subArraySumNonOverlappingSubArrays(nums: List[int], k: int) -> int:
| 27|     prefix_sum = collections.defaultdict(int)
| 28|     prefix_sum[0] = [-1]
| 31|     curr_sum = 0
| 32|     res = []
| 33|     for i in range(len(nums)):
| 34|         curr_sum += nums[i]
| 35|         if (curr_sum - k) in prefix_sum:
| 36|             res.append(*[nums[value+1:i+1] for value in prefix_sum[curr_sum-k]])
| 37|             prefix_sum.clear()
| 38|             prefix_sum[0] = []
| 39|         else:
| 40|             prefix_sum[curr_sum].append(i)
| 41|     return sum(list(map(sum, res)))

#====================================================================================================================================================
#       :: Arrays ::
#       :: Prefix Sum     [ TC: O(N) ; SC: O(N) ] ::
#       :: Three Pointers [ TC: O(N) ; SC: P(1) ] ::
#       :: array/binary_subarrays_with_sum.py ::
#       LC-930 | Binary Subarrays with Sum | https://leetcode.com/problems/binary-subarrays-with-sum/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given a binary array nums and an integer goal, return the number of non-empty subarrays with a sum goal.
|  3| 
|  4| A subarray is a contiguous part of the array.
|  5| 
|  6| Example 1:
|  7| 
|  8|   Input: nums = [1,0,1,0,1], goal = 2
|  9|   Output: 4
| 10|   Explanation: The 4 subarrays are bolded and underlined below:
| 11|   [1,0,1,0,1]
| 12|   [1,0,1,0,1]
| 13|   [1,0,1,0,1]
| 14|   [1,0,1,0,1]
| 15|
| 16| Example 2:
| 17| 
| 18|   Input: nums = [0,0,0,0,0], goal = 0
| 19|   Output: 15
| 20| 
| 21| Constraints:
| 22| 
| 23|   * 1 <= nums.length <= 3 * 104
| 24|   * nums[i] is either 0 or 1.
| 25|   * 0 <= goal <= nums.length
| 26| """
| 27|
| 28| """
| 29| Approach-1: Prefix Sum
| 30| ------------------------------------------------
| 31| Intuition
| 32| ------------------------------------------------
| 33| Let prefix_sum_array[i] = A[0] + A[1] + ... + A[i-1]. Then prefix_sum_array[j+1] - prefix_sum_array[i] = A[i] + A[i+1] + ... + A[j], the sum of the subarray [i, j].
| 34|
| 35| Hence, we are looking for the number of i < j with prefix_sum_array[j] - prefix_sum_array[i] = S.
| 36|
| 37| Algorithm
| 38| ------------------------------------------------
| 39| When you notice the keyword "subarrays" in a question, the idea of prefix sum should immediately pop in your head.
| 40|   - Brief definition of prefix sum: for i = 0...n-1, prefix_sum_array[i] = sum(A[:i+1]) => e.g for arr=[1, 0, 1], prefix_sum_array[2] = 1+0+1 = 2
| 41|   - So how do we utilize this to speed up our algorithm?
| 42|
| 43| ** Short answer: we count the occurrence of prefix sum. Why?
| 44|        
| 45| Consider the case where prefix_sum_array[j] - prefix_sum_array[i] = S. That means that A[i+1:j+1] == k.
| 46| Now lets take a look an example with arr=[1,0,1,0,1] and S = 2 
| 47|   - We also have i = 1, j = 4. 
| 48|   - With prefix_sum_array[1] = 1, prefix_sum_array[4] = 3, we have prefix_sum_array[j]-prefix_sum_array[i] = 2, which equals to  our specified S.
| 49| - That means -- when our algorithm is at index 4, we should add the occurence of acc-S (3-2 = 1) into our result,
| 50|   which accounts for the case [1,(0,1,0,1)] and [(1,0,1,0,1)].
| 51|   
| 52| - We do this for every iteration, and return the result.
| 53| """
| 54|
| 55| #TC: O(N), where N is the length of nums array.
| 56| #SC: O(N)
| 57| from typing import List
| 58| def numSubarraysWithSum(nums: List[int], goal: int) -> int:
| 59|     prefix_sum_array, cnt, prefix_sum = defaultdict(int), 0, 0
| 60|     prefix_sum_array[0] = 1
| 61|     for x in A:
| 62|         prefix_sum += x
| 63|         if prefix_sum-goal in prefix_sum_array:
| 64|             cnt += prefix_sum_array[prefix_sum-goal]
| 65|         prefix_sum_array[prefix_sum] += 1
| 66|     return cnt
| 67|
| 68| """
| 69| Approach 2: Three Pointers
| 70| ------------------------------------------------
| 71| Intuition
| 72| ------------------------------------------------
| 73| For each j, let's try to count the number of i's that have the subarray [i, j] equal to S.
| 74|
| 75| It is easy to see these i's form an interval [i_lo, i_hi], and each of i_lo, i_hi 
| 76| are increasing with respect to j. So we can use a "two pointer" style approach.
| 77|
| 78| Algorithm
| 79| ------------------------------------------------
| 80| For each j (in increasing order), let's maintain 3 variables:
| 81|
| 82| 1) temp_sum : to keep track of the current sum
| 83| 2) i_lo : Increase i_lo until temp_sum equals to the target sum
| 83| 3) i_hi : Increase i_hi starting from i_lo while nums[i_hi] is 0
| 84|
| 85| Then, (provided that temp_sum == goal), the number of subarrays ending in j is i_hi - i_lo + 1.
| 86|
| 87| As an example, with nums = [1,0,0,1,0,1] and goal = 2, when j = 5, we want i_lo = 1 and i_hi = 3.
| 88| """
| 89|
| 90| #TC: O(N), where N is the length of nums array.
| 91| #SC: O(1)
| 92| from typing import List
| 93| def numSubarraysWithSum(nums: List[int], goal: int) -> int:
| 94|     i_lo = i_hi = 0
| 95|     temp_sum = 0
| 96|     ans = 0
| 97|
| 98|     for j, x in enumerate(nums):
| 99|         temp_sum += x
|100|
|101|         # Maintain i_lo, temp_sum:
|102|         # Increase i_lo until temp_sum equals to the target sum
|103|         while i_lo < j and temp_sum > S:
|104|             temp_sum -= A[i_lo]
|105|             i_lo += 1
|106|
|107|         i_hi = i_lo
|108|
|109|         # Maintain i_hi, sum_hi:
|110|         # Increase i_hi starting from i_lo while A[i_hi] is 0
|111|         while i_hi < j and not A[i_hi]:
|112|             i_hi += 1
|113|
|114|         if temp_sum == S:
|115|             ans += i_hi - i_lo + 1
|116|
|117|     return ans

#====================================================================================================================================================
#       :: Arrays ::
#       :: Prefix Sum + Sliding Window ::
#       :: Two Sum + Dynamic Programming ::
#       :: array/number_of_submatrices_that_sum_to_target.py ::
#       LC-1074 | Number of Submatrices That Sum to Target | https://leetcode.com/problems/number-of-submatrices-that-sum-to-target/ | Hard
#       Hint (2D version of the problem => LC-560 | Subarray Sum Equals K | https://leetcode.com/problems/subarray-sum-equals-k/ | Medium )
#====================================================================================================================================================
|  1| """
|  2| Given a matrix and a target, return the number of non-empty submatrices that sum to target.
|  3|
|  4| A submatrix x1, y1, x2, y2 is the set of all cells matrix[x][y] with x1 <= x <= x2 and y1 <= y <= y2.
|  5|
|  6| Two submatrices (x1, y1, x2, y2) and (x1', y1', x2', y2') are different if they have some coordinate that is different: for example, if x1 != x1'.
|  7|
|  8| Example 1:
|  9| +-----+-----+-----+
| 10| |  0  |  1  |  0  |
| 11| +-----+-----+-----+
| 12| |  1  |  1  |  1  |
| 13| +-----+-----+-----+
| 14| |  0  |  1  |  0  |
| 15| +-----+-----+-----+
| 16| Input: matrix = [[0,1,0],[1,1,1],[0,1,0]], target = 0
| 17| Output: 4
| 18| Explanation: The four 1x1 submatrices that only contain 0.
| 19|
| 20| Example 2:
| 21|
| 22| Input: matrix = [[1,-1],[-1,1]], target = 0
| 23| Output: 5
| 24| Explanation: The two 1x2 submatrices, plus the two 2x1 submatrices, plus the 2x2 submatrix.
| 25|
| 26| Example 3:
| 27|
| 28| Input: matrix = [[904]], target = 0
| 29| Output: 0
| 30| """
| 31|
| 32| """
| 33| Approach-1: Prefix Sum + Sliding Window
| 34| ------------------------------------------------
| 35| For each row, calculate the prefix sum. For each pair of columns, calculate the sum of rows, using sliding window technique.
| 36| Now this problem is changed to problem 560 [ i.e., LC-560 | Subarray Sum Equals K ].
| 35| """
| 36|
| 37| # TC: O(N^2M)
| 38| # SC: O(N^2M)
| 39| # where, M = number of rows
| 40| #        N = number of columns
| 41| from typing import List
| 42| def numSubmatrixSumTarget(matrix: List[List[int]], target: int) -> int:
| 43|     # number of rows (i.e., m) and number of columns (i.e., n) of matrix
| 44|     m, n = len(matrix), len(matrix[0])
| 45|     # update prefix sum on each row
| 46|     for x in range(m):
| 47|         for y in range(n):
| 48|             matrix[x][y] = matrix[x][y] + matrix[x][y-1]
| 49|
| 50|     # number of submatrices that sum to target
| 51|     counter = 0
| 52|     # sliding windows on x-axis, in range [left, right]
| 53|     for left in range(n):
| 54|         for right in range(left, n):
| 55|             # accumulation of area so far
| 56|             accumulation = {0: 1}
| 57|             # area of current submatrices, bounded by [left, right] with height y
| 58|             area = 0
| 59|             # scan each possible row on y-axis
| 60|             for y in range(m):
| 61|                 if left > 0:
| 62|                     area += matrix[y][right] - matrix[y][left-1]
| 63|                 else:
| 64|                     area += matrix[y][right]
| 65|
| 66|                 # if ( area - target ) exist, then target must exist in submatrices
| 67|                 counter += accumulation.get(area - target, 0)
| 68|
| 69|                 # update dictionary with current accumulation area
| 70|                 accumulation[area] = accumulation.get(area, 0) + 1
| 71|
| 72|     return counter
| 73|
| 74| """
| 75| Approach-2: Two Sum + Dynamic Programming
| 76| ------------------------------------------------
| 77| Let us define by dp[i, j, k] sum of numbers in the rectangle i <= x < j and 0 <= y < m.
| 78| Why it is enough to evaluate only values on these matrices?
| 79| Because then we can use 2Sum problem [ i.e., i.e., LC-167 | Two Sum II - Input array is sorted ]: 
| 80|  * Any sum of elements in submatrix with coordinates a <= x < b and c <= y < d 
| 80|  * Can be evaluated as difference between sum of a <= x < b, 0 <= y < d and sum of a <= x < b, 0 <= y < c.
| 81|
| 82| So, let us fix a and b, and say we have sums S1, S2, ... Sm.
| 83| Then we want to find how many differences between these values give us our target.
| 84| The idea is to calculate cumulative sums and keep counter of values, and then check how many we have 
| 85| (we can not use sliding window, because we can have negative values),
| 86| see problem 560. Subarray Sum Equals K for more details.
| 87|
| 88| Algorithm
| 89| ------------------------------------------------
| 90| So, we have in total two stages of our algorithm:
| 91|
| 92|   1. Precompute all sums in rectangles of the type i <= x < j and 0 <= y < m.
| 93|   1. For each n*(n-1)/2 problems with fixed i and j, solve sum-problem in O(m) time.
| 94|
| 95| Complexity
| 96| ------------------------------------------------
| 97| Time complexity is O(n^2m), we need it for both stages. Space complexity is the same.
| 98| """
| 99| # TC: O(N^2M)
|100| # SC: O(N^2M)
|101| # where, M = number of rows
|102| #        N = number of columns
|103| import collections
|104| import itertools
|105| from typing import List
|106| def numSubmatrixSumTarget(matrix: List[List[int]], target: int) -> int:
|107|     m, n = len(matrix), len(matrix[0])
|108|     dp, ans = {}, 0
|109|     for k in range(m):
|110|         t = [0] + list(itertools.accumulate(matrix[k]))
|111|         for i, j in itertoools.combinations(range(n+1), 2):
|112|             dp[i, j, k] = dp.get((i,j,k-1), 0) + t[j] - t[i]
|113|
|114|     for i, j in itertools.combinations(range(n+1), 2):
|115|         T = collections.Counter([0])
|116|         for k in range(m):
|117|             ans += T[dp[i, j, k] - target]
|118|             T[dp[i, j, k]] += 1
|119|
|120|     return ans

#====================================================================================================================================================
#       :: Arrays ::
#       :: Sliding Windows ( Greedy Solution using Three Sliding Windows ) ::
#       :: array/number_of_submatrices_that_sum_to_target.py ::
#       LC-689 | Maximum Sum of 3 Non-Overlapping Subarrays | https://leetcode.com/problems/maximum-sum-of-3-non-overlapping-subarrays/ | Hard
#====================================================================================================================================================
|  1| """
|  2| Given an integer array nums and an integer k, find three non-overlapping subarrays of length k with maximum sum and return them.
|  3|
|  4| Return the result as a list of indices representing the starting position of each interval (0-indexed). If there are multiple answers, return the lexicographically smallest one.
|  5|
|  6| Example 1:
|  7|
|  8|   Input: nums = [1,2,1,2,6,7,5,1], k = 2
|  9|   Output: [0,3,5]
| 10|   Explanation: Subarrays [1, 2], [2, 6], [7, 5] correspond to the starting indices [0, 3, 5].
| 11|   We could have also taken [2, 1], but an answer of [1, 3, 5] would be lexicographically larger.
| 12|
| 13| Example 2:
| 14|
| 15|   Input: nums = [1,2,1,2,1,2,1,2,1], k = 2
| 16|   Output: [0,2,4]
| 17|
| 18| Constraints:
| 19|   * 1 <= nums.length <= 2 * 104
| 20|   * 1 <= nums[i] < 216
| 21|   * 1 <= k <= floor(nums.length / 3)
| 22| """
| 23|
| 24| """
| 25| 1. Convert nums to windows
| 26| ------------------------------------------------
| 27| That is, convert array of numbers into array of sum of windows. For example:
| 28|
| 29| nums = [1, 2, 3], w_size = 2
| 30| windows = [1 + 2, 2 + 3] = [3, 5]
| 31|
| 32| So the problem now is to choose 3 values from the window array and the difference of the indexes of these values must >=k
| 33|
| 34| 2. Define 3 arrays
| 35| ------------------------------------------------
| 36|   * take1[i]= biggest result we can get if only take 1 value from win[0] ... win[i]
| 37|   * take2[i]= biggest result we can get if only take 2 values from win[0] ... win[i]
| 38|   * take3[i]= biggest result we can get if only take 3 values from win[0] ... win[i]
| 39|
| 40| 3. Update 3 arrays dynamically
| 41| ------------------------------------------------
| 42|   * For take1, because only 1 window can be taken, so we just choose the current value or the previous value, depends on which one is bigger.
| 43|     So take1[i] = max(take1[i - 1], win[i])
| 44|
| 45|   * For take2, we need to select 2 windows, so we can either keep the previous selection, or we can take the current window and the biggest window in [0 ~ i - k], which is take1[i - k].
| 46|     So take2[i] = max(take2[i - 1], win[i] + take1[i - k])
| 47|
| 48|   * Same idea, take3[i] = max(take3[i - 1], win[i] + take2[i - k])
| 49|
| 50| 4. Remember the selection
| 51| ------------------------------------------------
| 52| If the problem is to return the biggest value, then step1~3 would be suffice, 
| 53| but we need to return the selection, so we need to remember not only the biggest value but also the index of selected window.
| 54|
| 55| Just modify to array to be:
| 56|
| 57|   take = [value, [indexes]]
| 58|
| 59| and when we update the value, we update the [indexes] too.
| 60|
| 61| """
| 62|
| 63| #TC: O(N)
| 64| #SC: O(N) [ Not, O(1) since slicing operations generate new arrays so it should be O(N) ]
| 65| from typing import List
| 66| def maxSumOfThreeSubarrays(nums: List[int], k: int) -> List[int]:
| 67|     take_1, take_2, take_3 = [(0,()) for _ in nums], [(0,()) for _ in nums], [(0,()) for _ in nums]
| 68|     sub_sum = sum(nums[:k])
| 69|     take_1[k-1] = (sub_sum, (0,))
| 70|        
| 71|     for i in range(k, len(nums)):
| 72|         sub_sum = sub_sum - nums[i-k] + nums[i]
| 73|         take_1[i] = max(take_1[i-1], (sub_sum,(i-k+1,)), key=lambda x:x[0])
| 74|         take_2[i] = max(take_2[i-1], (sub_sum + take_1[i-k][0], take_1[i-k][1] + (i-k+1,)), key=lambda x:x[0])
| 75|         take_3[i] = max(take_3[i-1], (sub_sum + take_2[i-k][0], take_2[i-k][1] + (i-k+1,)), key=lambda x:x[0])
| 76|        
| 77|     return take_3[-1][1]


Time complexity O(n^2 m^2 logm)
#====================================================================================================================================================
#       :: Arrays ::
#       :: Merge Sort ::
#       :: array/max_sum_of_rectangle_no_larger_than_k.py ::
#       LC-363 | Max Sum of Rectangle No Larger Than K | https://leetcode.com/problems/max-sum-of-rectangle-no-larger-than-k/ | Hard
#====================================================================================================================================================
|  1| """
|  2| Given an m x n matrix matrix and an integer k, return the max sum of a rectangle in the matrix such that its sum is no larger than k.
|  3|
|  4| It is guaranteed that there will be a rectangle with a sum no larger than k.
|  5|
|  6| Example 1:
|  7| +---------+---------+---------+
|  8| |         |+--------|--------+|
|  9| |    1    ||   0    |    1   ||
| 10| |         ||        |        ||
| 11| +---------+---------+---------+
| 12| |         ||        |        ||
| 13| |    0    ||  -2    |    3   ||    
| 14| |         |+-----------------+|
| 15| +---------+---------+---------+
| 16|   Input: matrix = [[1,0,1],[0,-2,3]], k = 2
| 17|   Output: 2
| 18|   Explanation: Because the sum of the blue rectangle [[0, 1], [-2, 3]] is 2, and 2 is the max number no larger than k (k = 2).
| 19|
| 20| Example 2:
| 21|
| 22|   Input: matrix = [[2,2,-1]], k = 3
| 23|   Output: 3
| 24|
| 25| """
| 26|
| 27| """
| 28| Approach-1 : Merge Sort
| 29| ------------------------------------------------
| 30| We can divide this question to two parts, the first is to build cumulative sum column by column, 
| 31| once we have this cumulative sum we have reduced the problem to finding 
| 32| maximum subarray sum less than or equal to K in 1-D array ( https://www.geeksforgeeks.org/maximum-sum-subarray-sum-less-equal-given-sum/ ), 
| 33| which solution can be found using mergesort which is O(NlogN) time complexity.
| 34| """
| 35|
| 36| # Approach-1: Solution using cumulative sum and merge-sort.
| 37| #TC: O( min(m,n)^2 * max(m,n) * log(max(m,n)) )
| 38| #SC: O( max(m,n) )
| 39| #where, m = number of rows ; n = number of columns
| 40| from typing import List
| 41| def maxSumSubmatrix(matrix: List[List[int]], k: int) -> int:
| 42|     m, n = len(matrix), len(matrix[0])
| 43|     M, N = min(m,n), max(m,n)
| 44|     ans = None
| 45|     def findMaxArea(sums, beg, end):
| 46|         if beg + 1 >= end: return None
| 47|         #mid = beg + ((end - beg)>>1)
| 48|         #mid = (beg + end) // 2
| 49|         mid = beg + ((end-beg)//2)
| 50|         res = max(findMaxArea(sums, beg, mid), findMaxArea(sums, mid, end))
| 51|         i = mid
| 52|         for l in sums[beg:mid]:
| 53|             while i < len(sums) and sums[i] - l <= k:
| 54|                 res = max(res, sums[i] - l)
| 55|                 i += 1
| 56|         sums[beg:end] = sorted(sums[beg:end])
| 57|         return res
| 58|
| 59|     for i1 in range(M):
| 60|         tmp = [0]*N
| 61|         for i2 in range(i1, M):
| 62|             sums, low, maxArea = [0], 0, None
| 63|             for j in range(N):
| 64|                 tmp[j] += matrix[i2][j] if m <= n else matrix[j][i2]
| 65|                 sums.append(sums[-1] + tmp[j])
| 66|                 maxArea = max(maxArea, sums[-1] - low)
| 67|                 low = min(low, sums[-1])
| 68|             if maxArea <= ans: continue
| 69|             if maxArea == k: return k
| 70|             if maxArea > k: maxArea = findMaxArea(sums, 0, N+1)
| 71|             ans = max(ans, maxArea)
| 72|     return ans or 0
| 73|
| 74| """
| 75| Approach-2 : Prefix Sum on 1D Array using Sorted Container
| 76| ------------------------------------------------
| 77| First of all, let us understand how to solve 1d problem: that is given list nums and number U 
| 78| we need to find the maximum sum of adjacent several elements such that its sum is no more than U.
| 79| Note, that it is very similar to problem 327 [ i.e., LC-327 | Count of Range Sum], but there the goal 
| 80| was to find not the biggest sum, but number of such sums.
| 81| However we can use the similar idea: let us add cumulative sums one by one,
| 82| that is if we have nums = [3, 1, 4, 1, 5, 9, 2, 6], then we add elements [3, 4, 8, 9, 14, 23, 25, 31].
| 83| Each time before we add element we do binary search of element s - U: the closest element bigger than s - U. If ind != len(SList), then we update our answer.
| 84|
| 85| When we found how to solve 1-d problem, it is time to work with 2-d problem.
| 86| Actually we need to solve O(m^2) 1-d problems, to choose numbers i,j such that 1 <= i <=j <= m.
| 87| What we can do is to calculate cumulative sums for each column and then for each pair create 
| 88| list of differences and apply our countRangeSum function.
| 89|
| 90| Complexity
| 91| ------------------------------------------------
| 92| Time complexity of 1-d problem is O(n log n), so time complexity of all algorithm is O(m^2*n log n).
| 93| It can be make O(n^2 * m log m) if we rotate our matrix, but in practice it works similar for me.
| 94| Space complexity is O(mn).
| 95| """
| 96|
| 97| # Approach-2: Prefix Sum on 1D Array using Sorted Container
| 98| # TC: O( m^2 * (n*log(n)) ) [ or , O( n^2 * (m*log(m)) ) if we rotate our matrix.
| 99| # SC: O( m*n )
|100| #where, m = number of rows ; n = number of columns
|101| import itertools
|102| from sortedcontainers import SortedList
|103| from typing import List
|104| def maxSumSubmatrix(matrix: List[List[int]], k: int) -> int:
|105|     def countRangeSum(nums, U):
|106|         SList, ans = SortedList([0]), -float("inf")
|107|         for s in itertools.accumulate(nums):
|108|             idx = SList.bisect_left(s - U) 
|109|             if idx < len(SList): ans = max(ans, s - SList[idx])        
|110|             SList.add(s)
|111|         return ans
|112|
|113|     m, n, ans = len(M), len(M[0]), -float("inf")
|114|        
|115|     for i, j in itertools.product(range(1, m), range(n)):
|116|         M[i][j] += M[i-1][j]
|117|
|118|     M = [[0]*n] + M
|119|
|120|     for r1, r2 in itertools.combinations(range(m + 1), 2):
|121|         row = [j - i for i, j in zip(M[r1], M[r2])]
|122|         ans = max(ans, countRangeSum(row, k))
|123|
|124|     return ans
|125|
|126| # Approach-3: Prefix Sum on 1D Array using Sorted Container
|127| # We can use slightly different function countRangeSum, where instead of SortedList we use usual list and insort function.
|128| # Complexity is O(n^2), however because n is not very big, it works even faster than previous method, like 2-3 times!
|129| # Complexity:
|130| # Time complexity is O(n^2*m^2), but with very small constant. Space complexity is O(mn).
|131| #
|132| # TC: O( n^2 * m^2 )
|133| # SC: O( m*n )
|134| #where, m = number of rows ; n = number of columns
|135| from bisect import bisect_left, insort
|136| from typing import List
|137| def maxSumSubmatrix(matrix: List[List[int]], k: int) -> int:
|138|     SList, ans = [0], -float("inf")
|139|     for s in accumulate(nums):
|140|         idx = bisect_left(SList, s - U) 
|141|         if idx < len(SList): ans = max(ans, s - SList[idx])        
|142|         bisect.insort(SList, s)
|143|     return ans

#====================================================================================================================================================
#       :: Arrays :: 
#       :: array/find_the_duplicate_number.py ::
#       LC-287 | Find the Duplicate Number | https://leetcode.com/problems/find-the-duplicate-number/ | Medium
#       Hint (Problem can be decomposed into find the starting point of loop in singly-linked list => i.e.,
#             LC-142 | Linked List Cycle II | https://leetcode.com/problems/linked-list-cycle-ii/ )
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
| 46| So the goal is to find loop in this linked list. Why there will be always loop? 
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
| 69| So now, the problem is to find the starting point of loop in singly-linked list 
| 70| (LC-142 : Linked List Cycle II),
| 71| which has a classical solution with two pointers: slow which moves one step at a time and fast, 
| 72| which moves two times at a time ( Floyd's Cycle Detection Algorithm ). To find this place 
| 73| we need to do it in two iterations: first we wait until fast pointer gains slow pointer 
| 74| and then we move slow pointer to the start and run them with the same speed and wait until they concide.
| 75|
| 76| Complexity: Time complexity is O(n), because we potentially can traverse all list.
| 77| Space complexity is O(1), because we actually do not use any extra space: our linked list is virtual.
| 78| """
| 79|
| 80| #Approach-1: Linked List Cycle (Floyd's Cycle Detection Algorithm)
| 81| #TC: O(N), because we potentially can traverse all list
| 82| #SC: O(1), because we actually do not use any extra space: our linked list is virtual
| 83| from typing import List
| 84| def findDuplicate(nums: List[int]) -> int:
| 85|     slow, fast = nums[0], nums[0]
| 86|     while True:
| 87|         slow, fast = nums[slow], nums[nums[fast]]
| 88|         if slow == fast: break
| 89| 
| 90|     slow = nums[0]
| 91|     while slow != fast:
| 92|         slow, fast = nums[slow], nums[fast]
| 93|     return slow
| 94|
| 95| """
| 96| Binary search solution
| 97| There is Binary Search solution with time complexity O(n log n) and space complexity O(1).
| 98| We have numbers from 1 to n. Let us choose middle element m = n//2 and count number of elements in list, which are less or equal than m.
| 99| If we have m+1 of them it means we need to search for duplicate in [1,m] range, else in [m+1,n] range.
|100| Each time we reduce searching range twice, but each time we go over all data. So overall complexity is O(n log n).
|101| """
|102|
|103| #Approach-1: Binary Search
|104| #TC: O(N*log(N))
|105| #SC: O(1), because we have numbers from 1 to N.
|106| from typing import List
|107| def findDuplicate(nums: List[int]) -> int:
|108|     beg, end = 1, len(nums)-1  
|109|     while beg + 1 <= end:
|110|         mid, count = (beg + end)//2, 0
|111|         for num in nums:
|112|             if num <= mid: count += 1        
|113|         if count <= mid:
|114|             beg = mid + 1
|115|         else:
|116|             end = mid
|117|     return end

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

