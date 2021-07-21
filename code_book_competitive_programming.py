# --- Code Book: Competitive Programming
# --- Source   : github.com/sm2774us/competitive_programming_book.git

#====================================================================================================================================================
#       :: Arrays ::
#       :: Kadane's Algorithm :: 
#       :: arrays/maximum_subarray.py ::
#       LC-53 | Maximum Subarray | https://leetcode.com/problems/maximum-subarrays/ | Easy
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
#       :: arrays/maximum_subarray_variant.py ::
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
#       :: arrays/product_of_array_except_self.py ::
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
#       :: arrays/continuous_subarray_sum.py ::
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
#       :: arrays/contiguous_array.py ::
#       LC-525 | Contiguous Array | https://leetcode.com/problems/contiguous-arrays/ | Medium
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
#       :: arrays/maximum_size_subarray_sum_equals_k.py ::
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
#       :: arrays/subarray_sum_divisible_by_k.py ::
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
#       :: arrays/make_sum_divisible_by_p.py ::
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
#       :: arrays/subarray_sum_equals_k.py ::
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
| 42|   * LC-525 - Contiguous Array                   : https://leetcode.com/problems/contiguous-arrays/
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
#       :: arrays/subarray_sum_equals_k_facebook_i_variant.py ::
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
#       :: arrays/subarray_sum_equals_k_non_overlapping_intervals_facebook_ii_variant.py ::
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
#       :: arrays/binary_subarrays_with_sum.py ::
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
#       :: arrays/number_of_submatrices_that_sum_to_target.py ::
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
#       :: arrays/maximum_sum_of_3_nonoverlapping_subarrays.py ::
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


#====================================================================================================================================================
#       :: Arrays ::
#       :: Merge Sort ::
#       :: Prefix Sum on 1D Array using Sorted Container ::
#       :: arrays/max_sum_of_rectangle_no_larger_than_k.py ::
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
|126| # Approach-3: Prefix Sum on 1D Array using Sorted Container ( using normal list and bisect.insort function )
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
#       :: Prefix Sum and DP ::
#       :: arrays/range_sum_query_immutable.py ::
#       LC-303 | Range Sum Query - Immutable | https://leetcode.com/problems/range-sum-query-immutable/ | Easy
#====================================================================================================================================================
|  1| """
|  2| Given an integer array nums, handle multiple queries of the following type:
|  3|
|  4|   1. Calculate the sum of the elements of nums between indices left and right inclusive where left <= right.
|  5|
|  6| Implement the NumArray class:
|  7|
|  8|   * NumArray(int[] nums) Initializes the object with the integer array nums.
|  9|   * int sumRange(int left, int right) Returns the sum of the elements of nums between indices left and right inclusive (i.e. nums[left] + nums[left + 1] + ... + nums[right]).
| 10|
| 11| Example 1:
| 12|
| 13|   Input:
| 14|   ["NumArray", "sumRange", "sumRange", "sumRange"]
| 15|   [[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]
| 16|   Output:
| 17|   [null, 1, -1, -3]
| 18|
| 19|   Explanation:
| 20|   NumArray numArray = new NumArray([-2, 0, 3, -5, 2, -1]);
| 21|   numArray.sumRange(0, 2); // return (-2) + 0 + 3 = 1
| 22|   numArray.sumRange(2, 5); // return 3 + (-5) + 2 + (-1) = -1
| 23|   numArray.sumRange(0, 5); // return (-2) + 0 + 3 + (-5) + 2 + (-1) = -3
| 24| """
| 25|
| 26| """
| 27| Hint:
| 28| ------------------------------------------------
| 29| #1) Think of the concept of dynamic programming, and look-up table.
| 30|
| 31| #2) Since input array, nums, is immutable, we can build a prefix sum table to speed up range sum query later.
| 32| ------------------------------------------------
| 33| Recurrence relationship:
| 34| ------------------------------------------------
| 35| Let S denotes the prefix sum table
| 36| S[ 0 ] = nums[ 0 ]
| 37| S[ i ] = S[ i - 1 ] + nums[ i ] for every i = 1, 2, 3, ..., n
| 38|
| 39| Range sum Query:
| 40| ------------------------------------------------
| 41| Query( i, j ) = S[ j ], if i =0
| 42| Query( i, j ) = S[ j ] - S[ i -1 ], otherwise.
| 43| 
| 44| Algorithm:
| 45| ------------------------------------------------
| 46| Step_#1: Build the prefix sum table based on recurrence relationship, during initialization in O(n).
| 47| Step_#2: Handle incoming range sum query by index lookup in prefix sum table, in O(1).
| 48|
| 49| References: [OpenGenius: Prefix sum array](https://iq.opengenus.org/prefix-sum-array/)
| 50| """
| 51|
| 52| # TC: O(1) time per query, O(n) time pre-computation. Since the prefix sum (i.e., cumulative sum) is cached, 
| 53| #     each sumRange query can be calculated in O(1) time.
| 54| # SC: O(n)
| 55| class NumArray:
| 56|
| 57|     def __init__(self, nums: List[int]):
| 58|         self.size = len(nums)
| 59|         if self.size:
| 60|             # build prefix sum table when input nums is valid
| 61|             self.prefix_sum = [ 0 for _ in range(self.size) ]
| 62|
| 63|             self.prefix_sum[0] = nums[0]
| 64|
| 65|             # prefix_Sum[k] = nums[0] + ... + nums[k]
| 66|             # prefix_Sum[k] = prefix_Sum[k-1] + nums[k]
| 67|             for k in range(1,self.size):
| 68|                 self.prefix_sum[k] = self.prefix_sum[k-1] + nums[k]
| 69|
| 70|    def sumRange(self, i: int, j: int) -> int:
| 71|        # reject query with invalid index
| 72|        if self.size == 0 or i < 0 or i > j or j >= self.size:
| 73|            return 0
| 74|
| 75|        # lookup table from prefix_Sum
| 76|        if i == 0:
| 77|            return self.prefix_sum[j]
| 78|        else:
| 79|            return self.prefix_sum[j]-self.prefix_sum[i-1]

#====================================================================================================================================================
#       :: Arrays ::
#       :: Prefix Sum and DP ::
#       :: arrays/range_sum_query_2d_immutable.py ::
#       LC-304 | Range Sum Query 2D - Immutable | https://leetcode.com/problems/range-sum-query-2d-immutable/ | Medium
#====================================================================================================================================================
|  1| """
|  1| Given a 2D matrix matrix, handle multiple queries of the following type:
|  2|
|  3| Calculate the sum of the elements of matrix inside the rectangle defined by its upper left corner (row1, col1) 
|  4| and lower right corner (row2, col2).
|  5|
|  6| Implement the NumMatrix class:
|  7|
|  8|   * NumMatrix(int[][] matrix) Initializes the object with the integer matrix matrix.
|  9|   * int sumRegion(int row1, int col1, int row2, int col2) Returns the sum of the elements of matrix inside the rectangle 
| 10|     defined by its upper left corner (row1, col1) and lower right corner (row2, col2).
| 11|
| 12| Example 1 :
| 13| ------------------------------------------------
| 14|   Input:
| 15|   ------------------------------------------------
| 16|   ["NumMatrix", "sumRegion", "sumRegion", "sumRegion"]
| 17|   [[[[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]], [2, 1, 4, 3], [1, 1, 2, 2], [1, 2, 2, 4]]
| 18|   Output:
| 19|   [null, 8, 11, 12]
| 20|
| 21|   Explanation:
| 22|   ------------------------------------------------
| 23|   NumMatrix numMatrix = new NumMatrix([[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]);
| 24|   numMatrix.sumRegion(2, 1, 4, 3); // return 8 (i.e sum of the red rectangle) 
| 25|   numMatrix.sumRegion(1, 1, 2, 2); // return 11 (i.e sum of the green rectangle)
| 26|   numMatrix.sumRegion(1, 2, 2, 4); // return 12 (i.e sum of the blue rectangle)
| 27| """
| 28|
| 29| """
| 30| The sum of the rectangle `(0,0)->(i,j)` is equal to the cell `(i,j)`,
| 32| plus the rectangle `(0,0)->(i,j-1)`, plus the rectangle `(0,0)->(i-1,j)`, 
| 33| minus the rectangle `(0,0)->(i-1,j-1)`. We subtract the last rectangle 
| 34| because it represents the overlap of the previous two rectangles that were added.
| 35|
| 36| With this information, we can use a dynamic programming (DP) approach to build 
| 37| a prefix sum matrix (dp) from M iteratively, where dp[i][j] will represent 
| 38| the sum of the rectangle (0,0)->(i,j). We'll add an extra row and column in order 
| 39| to prevent out-of-bounds issues at i-1 and j-1 (similar to a prefix sum array), and we'll fill dp with 0s.
| 40|
| 41| At each cell, we'll add its value from M to the dp values of the cell on the left 
| 42| and the one above, which represent their respective rectangle sums, and then subtract 
| 43| from that the top-left diagonal value, which represents the overlapping rectangle of the previous two additions.
| 44|
| 45| Then, we just reverse the process for sumRegion(): We start with the sum at dp[R2+1][C2+1]
| 46| (due to the added row/column), then subtract the left and top rectangles before adding back 
| 47| in the doubly-subtracted top-left diagonal rectangle.
| 48|
| 49| (Note: Even though the test cases will pass when using an int matrix for dp,
| 50| the values of dp can range from -4e9 to 4e9 per the listed constraints,
| 51| so we should use a data type capable of handling more than 32 bits.)
| 52|
| 53| References: [Solution: Range Sum Query 2D - Immutable](https://dev.to/seanpgallivan/solution-range-sum-query-2d-immutable-9ic)
| 54| """
| 55|
| 56| # Time Complexity:
| 57| # ------------------------------------------------
| 58| # constructor: O(M * N) where M and N are the dimensions of the input matrix
| 59| #
| 60| # sumRegion: O(1)
| 61| #
| 62| # Space Complexity:
| 63| # ------------------------------------------------
| 64| # constructor: O(M * N) for the DP matrix
| 65| # constructor: O(1) if you're able to modify the input and use an in-place DP approach
| 66| #
| 67| # sumRegion: O(1)
| 68| from typing import List
| 69| # TC: O(M*N)
| 70| # SC: O(1)
| 71| class NumMatrix:
| 72|
| 73|     def __init__(self, matrix: List[List[int]]):
| 74|         if matrix is None or not matrix:
| 75|             return
| 76|         rows, cols = len(matrix), len(matrix[0])
| 77|         self.sums = [ [0 for j in range(cols+1)] for i in range(rows+1) ]
| 78|         for i in range(1, rows+1):
| 79|             for j in range(1, cols+1):
| 80|                 self.sums[i][j] = matrix[i-1][j-1] + self.sums[i][j-1] + self.sums[i-1][j] - self.sums[i-1][j-1]
| 81|    
| 82|     def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
| 83|         row1, col1, row2, col2 = row1+1, col1+1, row2+1, col2+1
| 84|         #print(f'self.sums[row2][col2] = {self.sums[row2][col2]}')
| 85|         #print(f'self.sums[row2][col1-1] = {self.sums[row2][col1-1]}')
| 86|         #print(f'self.sums[row1-1][col2] = {self.sums[row1-1][col2]}')
| 87|         #print(f'self.sums[row1-1][col1-1] = {self.sums[row1-1][col1-1]}')
| 88|         return self.sums[row2][col2] - self.sums[row2][col1-1] - self.sums[row1-1][col2] + self.sums[row1-1][col1-1]
| 89|
| 90| # [More Elegant Solution](https://leetcode.com/problems/range-sum-query-2d-immutable/discuss/1204283/Python-short-dp-explained)
| 91| from itertools import product
| 92| from functools import lru_cache
| 93| from typing import List
| 94| # TC: O(M*N)
| 95| # SC: O(1)
| 96| class NumMatrix:
| 97|     def __init__(self, matrix: List[List[int]]):
| 98|     M, N = len(matrix), len(matrix[0])
| 99|     self.dp = [[0] * (N+1) for _ in range(M+1)] 
|100|     for c, r in product(range(N), range(M)):
|101|         self.dp[r+1][c+1] = matrix[r][c] + self.dp[r+1][c] + self.dp[r][c+1] - self.dp[r][c]
|102|    
|103|     @lru_cache(None)
|104|     def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
|105|         print(f'self.dp[r2+1][c2+1] = {self.dp[r2+1][c2+1]}')
|106|         print(f'self.dp[r1][c2+1] = {self.dp[r1][c2+1]}')
|107|         print(f'self.dp[r2+1][c1] = {self.dp[r2+1][c1]}')
|108|         print(f'self.dp[r1][c1] = {self.dp[r1][c1]}')
|109|         return self.dp[r2+1][c2+1] - self.dp[r1][c2+1] - self.dp[r2+1][c1] + self.dp[r1][c1]

#====================================================================================================================================================
#       :: Arrays ::
#       :: Prefix Sum and Integral Image ::
#       :: arrays/matrix_block_sum.py ::
#       LC-1314 | Matrix Block Sum | https://leetcode.com/problems/matrix-block-sum/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given a m x n matrix mat and an integer k, return a matrix answer where each answer[i][j] is the sum of all elements mat[r][c] for:
|  3|
|  4| i - k <= r <= i + k,
|  5| j - k <= c <= j + k, and
|  6| (r, c) is a valid position in the matrix.
|  7|
|  8| Example 1:
|  9|
| 10|   Input: mat = [[1,2,3],[4,5,6],[7,8,9]], k = 1
| 11|   Output: [[12,21,16],[27,45,33],[24,39,28]]
| 12|
| 13| Example 2:
| 14|
| 15|   Input: mat = [[1,2,3],[4,5,6],[7,8,9]], k = 2
| 16|   Output: [[45,45,45],[45,45,45],[45,45,45]]
| 17|
| 18| Note:
| 19| 
| 20|   rangeSum[i + 1][j + 1] corresponds to cell (i, j);
| 21|   rangeSum[0][j] and rangeSum[i][0] are all dummy values, which are used for the convenience of computation of DP state transmission formula.
| 22|
| 23| To calculate rangeSum, the ideas are as below
| 24|
| 25| +-----+-+-------+     +-------+------+     +-----+---------+     +-----+--------+     +-----+-+------+ 
| 26| |     | |       |     |       |      |     |     |         |     |     |        |     |              |
| 27| |     | |       |     |       |      |     |     |         |     |     |        |     |              |
| 28| +-----+-+       |     +-------+      |     |     |         |     +-----+        |     +     +-+      |
| 29| |     | |       |  =  |              |  +  |     |         |  -  |              |  +  |     | |      | 
| 30| +-----+-+       |     |              |     +-----+         |     |              |     +     +-+      |
| 31| |               |     |              |     |               |     |              |     |              |
| 32| |               |     |              |     |               |     |              |     |              |
| 33| +---------------+     +--------------+     +---------------+     +--------------+     +--------------+
| 34|
| 35| rangeSum[i+1][j+1] =  rangeSum[i][j+1] + rangeSum[i+1][j]    -   rangeSum[i][j]    +  mat[i][j]
| 36|
| 37| So, we use the same idea to find the specific block's sum.
| 38|
| 39| +---------------+   +--------------+   +---------------+   +--------------+   +--------------+
| 40| |               |   |         |    |   |   |           |   |         |    |   |   |          |
| 41| |   (r1,c1)     |   |         |    |   |   |           |   |         |    |   |   |          |
| 42| |   +------+    |   |         |    |   |   |           |   +---------+    |   +---+          |
| 43| |   |      |    | = |         |    | - |   |           | - |      (r1,c2) | + |   (r1,c1)    |
| 44| |   |      |    |   |         |    |   |   |           |   |              |   |              |
| 45| |   +------+    |   +---------+    |   +---+           |   |              |   |              |
| 46| |        (r2,c2)|   |       (r2,c2)|   |   (r2,c1)     |   |              |   |              |
| 47| +---------------+   +--------------+   +---------------+   +--------------+   +--------------+
| 48|
| 49| If the above is still not clear enough for you, the website below explains the concept really well:
| 50| References: 
| 51| https://leetcode.com/problems/matrix-block-sum/discuss/482730/Python-O(-m*n-)-sol.-by-integral-image-technique.-90%2B-with-Explanation
| 52| https://computersciencesource.wordpress.com/2010/09/03/computer-vision-the-integral-image/
| 53|
| 54| Analysis:
| 55|
| 56| Time & space: O(m * n).
| 57| """
| 58|
| 59| # Prefix Sum and Integral Image
| 60| # TC: O(m * n)
| 61| # SC: O(m * n)
| 62| from collections import defaultdict
| 63| from typing import List
| 64| class Solution(object):
| 65|     def matrixBlockSum(self, mat: List[List[int]], k: int) -> List[List[int]]:
| 66|         n = len(mat)
| 67|         m = len(mat[0])
| 68|         sums = defaultdict(int)
| 69|
| 70|         for i in range(n):
| 71|             for j in range(m):
| 72|                 sums[i,j] = sums[i-1,j] + sums[i,j-1] - sums[i-1,j-1] + mat[i][j]
| 73|         result = [[0] * m for _ in range(n)]
| 74|         for i in range(n):
| 75|             for j in range(m):
| 76|                 r1 = max(0, i-K)
| 77|                 c1 = max(0, j-K)
| 78|                 r2 = min(n-1, i+K)
| 79|                 c2 = min(m-1, j+K)
| 80|                 result[i][j] = sums[r2,c2] - sums[r2,c1-1] - sums[r1-1,c2] + sums[r1-1,c1-1]
| 81|         return result

#====================================================================================================================================================
#       :: Arrays ::
#       :: Sliding Window ::
#       :: arrays/count_number_of_nice_subarrays.py ::
#       LC-1248 | Count Number of Nice Subarrays | https://leetcode.com/problems/count-number-of-nice-subarrays/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given an array of integers nums and an integer k. A continuous subarray is called nice if there are k odd numbers on it.
|  3| 
|  4| Return the number of nice sub-arrays.
|  5| 
|  6| Example 1:
|  7|
|  8|   Input: nums = [1,1,2,1,1], k = 3
|  9|   Output: 2
| 10|   Explanation: The only sub-arrays with 3 odd numbers are [1,1,2,1] and [1,2,1,1].
| 11|
| 12| Example 2:
| 13|
| 14|   Input: nums = [2,4,6], k = 1
| 15|   Output: 0
| 16|   Explanation: There is no odd numbers in the array.
| 17|
| 18| Example 3:
| 19|
| 20|   Input: nums = [2,2,2,1,2,2,1,2,2,2], k = 2
| 21|   Output: 16
| 22| """
| 23|
| 24| """
| 25| 1. Whenever the count of odd numbers reach k, for each high boundary of the sliding window, we have indexOfLeftMostOddInWin - lowBound options for the low boundary, where indexOfLeftMostOddInWin is the index of the leftmost odd number within the window, and lowBound is the index of the low boundary exclusively;
| 26| 2. Whenever the count of odd numbers more than k, shrink the low boundary so that the count back to k
| 27| """
| 28|
| 29| # TC: O(N)
| 30| # SC: O(1)
| 31| from typing import List
| 32| def numberOfSubarrays(nums: List[int], k: int) -> int:
| 33|     low_bound, index_of_left_most_odd_in_win, ans = -1, 0, 0
| 34|     for num in nums:
| 35|         k -= num % 2
| 36|         if nums[index_of_left_most_odd_in_win] % 2 == 0:
| 37|             index_of_left_most_odd_in_win += 1
| 38|         if k < 0:
| 39|             low_bound = index_of_left_most_odd_in_win
| 40|         while k < 0:    
| 41|             index_of_left_most_odd_in_win += 1
| 42|             k += nums[index_of_left_most_odd_in_win] % 2
| 43|         if k == 0:
| 44|             ans += index_of_left_most_odd_in_win - low_bound
| 45|     return ans

#====================================================================================================================================================
#       :: Arrays ::
#       :: MONOTONOUS STACK ::
#       :: arrays/largest_rectangle_in_histogram.py ::
#       LC-84 | Largest Rectangle in Histogram | https://leetcode.com/problems/largest-rectangle-in-histogram/ | Hard
#====================================================================================================================================================
|  1| """
|  2| Given an array of integers heights representing the histogram's bar height where the width of each bar is 1,
|  3| return the area of the largest rectangle in the histogram.
|  4| 
|  5| Example 1:
|  6|
|  7|   Input: heights = [2,1,5,6,2,3]
|  8|   Output: 10
|  9|   Explanation: The above is a histogram where width of each bar is 1.
| 10|   The largest rectangle is shown in the red area, which has an area = 10 units.
| 11|
| 12| Example 2:
| 13|
| 14|   Input: heights = [2,4]
| 15|   Output: 4
| 16| """
| 17|
| 18| """
| 19| Definition of Monotonic Queue
| 20| ------------------------------------------------
| 21| A monotonic Queue is a data structure the elements from the front to the end is strictly either increasing or decreasing. 
| 22|
| 23|   * Monotonic increasing queue: to push an element e, starts from the rear element, we pop out element s≥e(violation);
| 24|   * Monotonic decreasing queue: we pop out element s<=e (violation).
| 25|   * Sometimes, we can relax the strict monotonic condition, and can allow the stack or queue have repeat value.
| 26|
| 27| Features and Basic Code
| 28| ------------------------------------------------
| 29| To get the feature of the monotonic queue, with [5, 3, 1, 2, 4] as example, if it is increasing:
| 30|
| 31| index   v   Increasing queue        Decreasing queue
| 32| 1       5   [5]                     [5]
| 33| 2       3   [3] 3 kick out 5        [5, 3] #3->5
| 34| 3       1   [1] 1 kick out 3        [5, 3, 1] #1->3
| 35| 4       2   [1, 2] #2->1            [5, 3, 2] 2 kick out 1 #2->3
| 36| 5       4   [1, 2, 4] #4->2         [5,4] 4 kick out 2, 3 #4->2
| 37|
| 38| The features can be generalized:
| 39| ------------------------------------------------
| 40|   * increasing queue: find the first element smaller than current either in the left (from pushing in) or in the right (from popping out);
| 41|   * decreasing queue: find the first element larger than current either in the left (from pushing in) or in the right (from popping out);
| 42|
| 43| This monotonic queue is actually a data structure that needed to add/remove element from the end.
| 44| In some application we might further need to remove element from the front.
| 45| Thus Deque from collections fits well to implement this data structure.
| 46| """
| 47|
| 48| """
| 49| ## LOGIC ##
| 50| 1. Before Solving this problem, go through Monotone stack.
| 51| 2. Using Monotone Stack we can solve: 
| 52|    a) Next Greater Element
| 53|    b) Next Smaller Element
| 54|    c) Prev Greater Element
| 55|    d) Prev Smaller Element
| 56| 3. Using 'NSE' (Next Smallest Element ) Monotone Stack concept, we can find width of rectangles,
| 57|    height obviously will be the minimum of those. Thus we can calculate the area
| 58| 4. As we are using NSE concept, adding 0 to the end, will make sure that stack is EMPTY at the end. 
| 59|    ( so all the areas can be calculated while popping )
| 60|
| 61| """
| 62|
| 63| # TC: O(n)
| 64| # SC: O(n)
| 65| # where n is number of bars
| 66| from typing import List
| 67| def largestRectangleArea(heights: List[int]) -> int:
| 68|     ## RC ##
| 69|     ## APPROACH : MONOTONOUS INCREASING STACK ##
| 70|     ## Similar to Leetcode: 1475. Final Prices With a Special Discount in a Shop ##
| 71|     ## Similar to Leetcode: 907. Sum Of Subarray Minimums ##
| 72|     ## Similar to Leetcode: 85. maximum Rectangle ##
| 73|     ## Similar to Leetcode: 402. Remove K Digits ##
| 74|     ## Similar to Leetcode: 456. 132 Pattern ##
| 75|     ## Similar to Leetcode: 1063. Number Of Valid Subarrays ##
| 76|     ## Similar to Leetcode: 739. Daily Temperatures ##
| 77|     ## Similar to Leetcode: 1019. Next Greater Node In LinkedList ##
| 78|        
| 79|        
| 80|     heights.append(0)
| 81|     stack = [-1]
| 82|     ans = 0
| 83|     for i in range(len(heights)):
| 84|         while heights[i] < heights[stack[-1]]:
| 85|             height = heights[stack.pop()]
| 86|             width = i - stack[-1] - 1
| 87|             ans = max(ans, height * width)
| 88|         stack.append(i)
| 89|     heights.pop()
| 90|     return ans        

#====================================================================================================================================================
#       :: Arrays ::
#       :: MONOTONOUS STACK ::
#       :: arrays/maximal_rectangle.py ::
#       LC-85 | Maximal Rectangle | https://leetcode.com/problems/maximal-rectangle/ | Hard
#====================================================================================================================================================
|  1| """
|  2| Given a rows x cols binary matrix filled with 0's and 1's, find the largest rectangle containing only 1's and return its area.
|  3|
|  4| Example 1:
|  5|
|  6|   Input: matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
|  7|   Output: 6
|  8|   Explanation: The maximal rectangle is shown in the above picture.
|  9|
| 10| Example 2:
| 11|
| 12|   Input: matrix = []
| 13|   Output: 0
| 14|
| 15| Example 3:
| 16|
| 17|   Input: matrix = [["0"]]
| 18|   Output: 0
| 19|
| 20| Example 4:
| 21|
| 22|   Input: matrix = [["1"]]
| 23|   Output: 1
| 24|
| 25| Example 5:
| 26|
| 27|   Input: matrix = [["0","0"]]
| 28|   Output: 0
| 29| """
| 30|
| 31| """
| 32| This problem can be visualised as an extension to largest rectangle in histogram
| 33| In the above problem you can use Next Smallest Element (NSE) logic 
| 34| on a monotonically increasing stack to find the maximum rectangle possible in the histogram given.
| 35|
| 36| You can easily extend that concept on this 2D matrix by 
| 37| constructing/updating histogram row by row and processing on the histogram.
| 38|
| 39|   1. For the first row the whole row is a histogram with bar heights either 1 or 0
| 40|   1. For the next row we update the histogram as
| 41|      => If the value of current grid is '0' then histogram height is reset to 0 (as the continuation of the rectangle breaks here)
| 42|      => If the value of the current grid is '1' then histogram height can be increased by 1 vertically
| 43|   1. So on for rest of the rows...
| 44|      Note: As we are processing row by row so rectangle continuity in horizontal direction is handled 
| 45|      in histogram automatically while the vertical continuity is handled by updating the histogram as in step 2.
| 46|   1. Now after updating the histogram for each row, we calculate the maximum area of rectangle possible.
| 47|   1. Once we have maximum from all the rows, we pick the maximum of the maximums to get our final answer.
| 48|
| 49| Example:
| 50| Input:
| 51|
| 52|   [
| 53|     ["1","0","1","0","0"],
| 54|     ["1","0","1","1","1"],
| 55|     ["1","1","1","1","1"],
| 56|     ["1","0","0","1","0"]
| 57|   ]
| 58|
| 59| Histogram creation and updation for each row of matrix:
| 60|
| 61|   Row 1 => [1, 0, 1, 0, 0]   [current max = 1 and overall max = 1]
| 62|   Row 2 => [2, 0, 2, 1, 1]   [current max = 3 and overall max = 3]
| 63|   Row 3 => [3, 1, 3, 2, 2]   [current max = 6 and overall max = 6]
| 64|   Row 4 => [4, 0, 0, 3, 0]   [current max = 4 and overall max = 6]
| 65|
| 66| As you can see the final answer will be picked as 6
| 67| """
| 68|
| 69| # TC: O(R * C)
| 70| # SC: O(C)
| 71| # where, R = number of rows and C = number of columns
| 72| from typing import List
| 73| class Solution:
| 74|     def maximalRectangle(self, matrix: List[List[str]]) -> int:
| 75|         if not matrix:
| 76|             return 0  # To handle `matrix = []` case
| 77|
| 78|         # Prepare histogram for the first row of input matrix and find maximum area.
| 79|         histogram = list(map(int, matrix[0]))
| 80|         histogram.append(0)
| 81|         current_max = self.max_rect_histogram(histogram)
| 82|		
| 83|         # Process on the remaining rows of matrix
| 84|         for row in matrix[1:]:
| 85|             for i in range(len(row)):
| 86|                 # update histogram, if grid is a '0' reset histogram height else increase height by one
| 87|                 if row[i] == '0':
| 88|                     histogram[i] = 0
| 89|                 else:
| 90|                     histogram[i] += int(row[i])
| 91|             # Once histogram is updated for current row find the maximum rectangle area in current histogram
| 92|             current_max = max(current_max, self.max_rect_histogram(histogram))
| 93|         return current_max
| 94|    
| 95|     def max_rect_histogram(self, histogram):
| 96|         stack = [-1]
| 97|         mx = 0
| 98|         for i,v in enumerate(histogram):
| 99|             # As long as the stack has increasing value keep adding the index of histogram
|100|             # If the insertion to stack will result in a decreasing stack, then keep poping till it becomes increasing again
|101|             # For each pop, calculate the area of the rectangle
|102|             while(stack[-1] != -1 and histogram[stack[-1]] > v):
|103|                 height = histogram[stack.pop()]
|104|                 # i is right limit, stack[-1] is left limit so width of rectangle in consideration is r-l-1
|105|                 width = i - stack[-1] - 1
|106|                 mx = max(mx, height*width)
|107|             stack.append(i)
|108|         return mx
        
#====================================================================================================================================================
#       :: Arrays ::
#       :: MONOTONOUS STACK ::
#       :: arrays/final_prices_with_a_special_discount_in_a_shop.py ::
#       LC_1475 | Final Prices With a Special Discount in a Shop | https://leetcode.com/problems/final-prices-with-a-special-discount-in-a-shop/ | Easy
#====================================================================================================================================================
|  1| """
|  2| Given the array prices where prices[i] is the price of the ith item in a shop. 
|  3| There is a special discount for items in the shop, if you buy the ith item, 
|  4| then you will receive a discount equivalent to prices[j] 
|  5| where j is the minimum index such that j > i and prices[j] <= prices[i], otherwise, you will not receive any discount at all.
|  6|
|  7| Return an array where the ith element is the final price you will pay for the ith item of the shop considering the special discount.
|  8|
|  9| 
| 10| Example 1:
| 11|
| 12|   Input: prices = [8,4,6,2,3]
| 13|   Output: [4,2,4,2,3]
| 14|   Explanation: 
| 15|   For item 0 with price[0]=8 you will receive a discount equivalent to prices[1]=4, therefore, the final price you will pay is 8 - 4 = 4. 
| 16|   For item 1 with price[1]=4 you will receive a discount equivalent to prices[3]=2, therefore, the final price you will pay is 4 - 2 = 2. 
| 17|   For item 2 with price[2]=6 you will receive a discount equivalent to prices[3]=2, therefore, the final price you will pay is 6 - 2 = 4. 
| 18|   For items 3 and 4 you will not receive any discount at all.
| 19|
| 20| Example 2:
| 21|
| 22|   Input: prices = [1,2,3,4,5]
| 23|   Output: [1,2,3,4,5]
| 24|   Explanation: In this case, for all items, you will not receive any discount at all.
| 25|
| 26| Example 3:
| 27|
| 28|   Input: prices = [10,1,1,6]
| 29|   Output: [9,0,1,6]
| 30| """
| 31|
| 32| """
| 33| MONOTONOUS STACK
| 34| ------------------------------------------------
| 34| In short, use a stack to maintain the indices of strickly increasing prices.
| 35|
| 36|   1. Clone the prices array as original price before discount;
| 37|   2. Use a stack to hold the indices of the previous prices that are less than current price;
| 38|   3. Keep poping out the prices that are NO less than current price, deduct current price as discount from previous prices.
| 39| """
| 40|
| 41| # TC: O(n)
| 42| # SC: O(n)
| 43| from typing import List
| 44| def finalPrices(prices: List[int]) -> List[int]:
| 45|     res, stack = prices[:], []
| 46|     for i, price in enumerate(prices):
| 47|         while stack and prices[stack[-1]] >= price:
| 48|             res[stack.pop()] -= price
| 49|         stack.append(i)
| 50|     return res

#====================================================================================================================================================
#       :: Arrays ::
#       :: MONOTONOUS STACK ::
#       :: arrays/sum_of_subarray_minimums.py ::
#       LC_907 | Sum of Subarray Minimums | https://leetcode.com/problems/sum-of-subarray-minimums/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given an array of integers arr, find the sum of min(b), where b ranges over every (contiguous) subarray of arr.
|  3| Since the answer may be large, return the answer modulo 10^9 + 7.
|  4|
|  5| Example 1:
|  6|
|  7|   Input: arr = [3,1,2,4]
|  8|   Output: 17
|  9|   Explanation: 
| 10|   Subarrays are [3], [1], [2], [4], [3,1], [1,2], [2,4], [3,1,2], [1,2,4], [3,1,2,4]. 
| 11|   Minimums are 3, 1, 2, 4, 1, 1, 2, 1, 1, 1.
| 12|   Sum is 17.
| 13|
| 14| Example 2:
| 15|
| 16|   Input: arr = [11,81,94,43,3]
| 17|   Output: 444
| 18|
| 19| """
| 20| """
| 22| Monotonous Stack ( or Monotone Stack ) :
| 23| ------------------------------------------------
| 24| Before diving into the solution, we first introduce a very important stack type, which is called monotone stack .
| 25|
| 26| What is monotonous increase stack?
| 27| Roughly speaking, the elements in the an monotonous increase stack keeps an increasing order.
| 28|
| 29| The typical paradigm for monotonous increase stack:
| 30|
| 31| for(int i = 0; i < A.size(); i++){
| 32|     while(!in_stk.empty() && in_stk.top() > A[i]){
| 33|         in_stk.pop();
| 34|     }
| 35|     in_stk.push(A[i]);
| 36| }
| 37|
| 38| What can monotonous increase stack do?
| 39| ------------------------------------------------
| 40| (1) find the previous less element of each element in a vector with O(n) time:
| 41|     * What is the previous less element of an element?
| 42|       For example:
| 43|       [3, 7, 8, 4]
| 44|       The previous less element of 7 is 3.
| 45|       The previous less element of 8 is 7.
| 46|       The previous less element of 4 is 3.
| 47|       ------------------------------------
| 48|       There is no previous less element for 3.
| 49|
| 50| For simplicity of notation, we use abbreviation PLE to denote Previous Less Element.
| 51|                                                 ---           -        -    -
| 52|     * C++ code (by slitghly modifying the paradigm):
| 53|       Instead of directly pushing the element itself, here for simplicity, we push the index.
| 54|       We do some record when the index is pushed into the stack.
| 55|
| 56| // previous_less[i] = j means A[j] is the previous less element of A[i].
| 57| // previous_less[i] = -1 means there is no previous less element of A[i].
| 58| vector<int> previous_less(A.size(), -1);
| 59| for(int i = 0; i < A.size(); i++){
| 60|     while(!in_stk.empty() && A[in_stk.top()] > A[i]){
| 61|         in_stk.pop();
| 62|     }
| 63|     previous_less[i] = in_stk.empty()? -1: in_stk.top();
| 64|     in_stk.push(i);
| 65| }
| 66|
| 67| (2) find the next less element of each element in a vector with O(n) time:
| 68|     * What is the next less element of an element?
| 69|       For example:
| 70|       [3, 7, 8, 4]
| 71|       The next less element of 8 is 4.
| 72|       The next less element of 7 is 4.
| 73|       There is no next less element for 3 and 4.
| 74|
| 75| For simplicity of notation, we use abbreviation NLE to denote Next Less Element.
| 76|
| 77|     * C++ code (by slighly modifying the paradigm):
| 78|       We do some record when the index is poped out from the stack.
| 79|
| 80| // next_less[i] = j means A[j] is the next less element of A[i].
| 81| // next_less[i] = -1 means there is no next less element of A[i].
| 82| vector<int> previous_less(A.size(), -1);
| 83| for(int i = 0; i < A.size(); i++){
| 84|     while(!in_stk.empty() && A[in_stk.top()] > A[i]){
| 85|         auto x = in_stk.top(); in_stk.pop();
| 86|         next_less[x] = i;
| 87|     }
| 88|     in_stk.push(i);
| 89| }
| 90| ================================================
| 91| How can the monotonous increase stack be applied to this problem?
| 92|
| 93| For example:
| 94| Consider the element 3 in the following vector:
| 95|
| 96|                       [2, 9, 7, 8, 3, 4, 6, 1]
| 97|			 |                    |
| 98|            the previous less       the next less 
| 99|               element of 3          element of 3
|100|
|101| After finding both NLE and PLE of 3, we can determine the
|102| distance between 3 and 2(previous less) , and the distance between 3 and 1(next less).
|103| In this example, the distance is 4 and 3 respectively.
|104|
|105| How many subarrays with 3 being its minimum value?
|106| The answer is 4*3.
|107|
|108| 9 7 8 3 
|109| 9 7 8 3 4 
|110| 9 7 8 3 4 6 
|111| 7 8 3 
|112| 7 8 3 4 
|113| 7 8 3 4 6 
|114| 8 3 
|115| 8 3 4 
|116| 8 3 4 6 
|117| 3 
|118| 3 4 
|119| 3 4 6
|120|
|121| How much the element 3 contributes to the final answer?
|122| ------------------------------------------------
|123| It is 3*(4*3).
|124| What is the final answer?
|125| ------------------------------------------------
|126| Denote by left[i] the distance between element A[i] and its PLE.
|127|                                                             ---
|128| Denote by right[i] the distance between element A[i] and its NLE.
|129|                                                              ---
|130|
|131| The final answer is,
|132| ------------------------------------------------
|133| sum(A[i]*left[i]*right[i] )
|134|
|135| """
|136|
|137| """
|138| Solution Approach:
|139| ------------------------------------------------
|140| Use a monotonous non-decreasing stack to store the left boundary and right boundary where a number is the minimal number 
|141| in the sub-array :
|142|
|143| e.g. given [3,1,2,4],
|144| For 3, the boudary is: | 3 | ...
|145| For 1, the boudray is: | 3 1 2 4 |
|146| For 2, the boudray is: ... | 2 4 |
|147| For 4, the boudary is: ... | 4 |
|148|
|149| The times a number n occurs in the minimums is |left_bounday-indexof(n)| * |right_bounday-indexof(n)|
|150|
|151| The total sum is sum([n * |left_bounday - indexof(n)| * |right_bounday - indexof(n)| for n in array])
|152|
|153| After a number n pops out from an increasing stack, the current stack top is n's left_boundary,
|154| the number forcing n to pop is n's right_boundary.
|155|
|156| A tricky here is to add MIN_VALUE at the head and end.
|157|
|158| Complexity:
|159| ------------------------------------------------
|160| All elements will be pushed twice and popped at most twice
|160| O(n) time, O(n) space
|161|
|162| """
|163|
|164| # TC: O(n)
|165| # SC: O(n)
|166| import math
|167| from typing import List
|168| def sumSubarrayMins(arr: List[int]) -> int:
|169|     res = 0
|170|     stack = []  #  non-decreasing 
|171|     A = [-math.inf] + A + [-math.inf]
|172|     for i, n in enumerate(A):
|173|         while stack and A[stack[-1]] > n:
|174|             cur = stack.pop()
|175|             res += A[cur] * (i - cur) * (cur - stack[-1]) 
|176|         stack.append(i)
|177|     return res % (10**9 + 7)

#====================================================================================================================================================
#       :: Arrays ::
#       :: MONOTONOUS STACK ::
#       :: arrays/online_stock_span.py ::
#       LC_901 | Online Stock Span | https://leetcode.com/problems/online-stock-span/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Write a class StockSpanner which collects daily price quotes for some stock,
|  3| and returns the span of that stock's price for the current day.
|  4|
|  5| The span of the stock's price today is defined as the maximum number of consecutive days (starting from today and going backwards) 
|  6| for which the price of the stock was less than or equal to today's price.
|  7|
|  8| For example, if the price of a stock over the next 7 days were [100, 80, 60, 70, 60, 75, 85],
|  9| then the stock spans would be [1, 1, 1, 2, 1, 4, 6].
| 10|
| 11| Example 1:
| 12|
| 13|   Input: ["StockSpanner","next","next","next","next","next","next","next"], [[],[100],[80],[60],[70],[60],[75],[85]]
| 14|   Output: [null,1,1,1,2,1,4,6]
| 15|   Explanation: 
| 16|   First, S = StockSpanner() is initialized.  Then:
| 17|   S.next(100) is called and returns 1,
| 18|   S.next(80) is called and returns 1,
| 19|   S.next(60) is called and returns 1,
| 20|   S.next(70) is called and returns 2,
| 21|   S.next(60) is called and returns 1,
| 22|   S.next(75) is called and returns 4,
| 23|   S.next(85) is called and returns 6.
| 24|
| 25|   Note that (for example) S.next(75) returned 4, because the last 4 prices
| 26|   (including today's price of 75) were less than or equal to today's price.
| 27|
| 28| """
| 29|
| 30| """
| 31| Solution Approach (Monotonic Stack) :
| 32| ------------------------------------------------
| 33| Push every pair of <price, result> to a stack.
| 34| Pop lower price from the stack and accumulate the count.
| 35|
| 36| One price will be pushed once and popped once.
| 37| So 2 * N times stack operations and N times calls.
| 38| 
| 39| Amortized complexity for each calculation is O(1).
| 40| You do n operations, so complexity is O(n).
| 41| 
| 42| Solution Visualization:
| 43| ------------------------------------------------
| 44| First, S = StockSpanner() is initial    +----------+  +----------+  +----------+  +----------+  +----------+  +----------+  +----------+
| 45|                                    __   |          |  |          |  |          |  |          |  | (60,<1>) |  |          |  |          |
| 46| S.next(100) is called and returns /1,\  +----------+  +----------+  +----------+  +----------+  +----------+  +----------+  +----------+
| 47|                                  /    | |          |  |          |  | (60,<1>) |  | (70,<2>) |  |  (70,2)  |  | (75,<4>) |  |          |
| 48| S.next(80) is called and returns | 1, | +----------+  +----------+  +----------+  +----------+  +----------+  +----------+  +----------+
| 49|                                  |    | |          |  | (80,<1>) |  |  (80,1)  |  |  (80,1)  |  |  (80,1)  |  |  (80,1)  |  | (85,<6>) |
| 50| S.next(60) is called and returns | 1  | +----------+  +----------+  +----------+  +----------+  +----------+  +----------+  +----------+
| 51|                                  |    | | (100,<1>)|  |  (100,1) |  |  (100,1) |  |  (100,1) |  |  (100,1) |  |  (100,1) |  |  (100,1) |
| 52| S.next(70) is called and returns | 2, | +----------+  +----------+  +----------+  +----------+  +----------+  +----------+  +----------+
| 53| S.next(60) is called and returns | 1, |
| 54| S.next(75) is called and returns | 4, | +----------------------------------------------------------------------------------------------->
| 55| S.next(85) is called and returns | 6, |                                    Execution Order
| 56|                                  \____/
| 57|
| 58| """
| 59| # TC: O(n)
| 60| # SC: O(n)
| 61| # faster solution without pop() 
| 62| import math
| 63| class StockSpanner:   
| 64|     def __init__(self):
| 65|         self.stack = [[math.inf, 1]]
| 66|
| 67|     def next(self, price: int) -> int:
| 68|         res = 1
| 69|         while price >= self.stack[-res][0]:
| 70|             res += self.stack[-res][1]
| 71|         self.stack.append([price, res])
| 72|         return res

#====================================================================================================================================================
#       :: Arrays ::
#       :: MONOTONOUS STACK ::
#       :: arrays/find_the_most_competitive_subsequence.py ::
#       LC_1673 | Find the Most Competitive Subsequence | https://leetcode.com/problems/find-the-most-competitive-subsequence/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given an integer array nums and a positive integer k, return the most competitive subsequence of nums of size k.
|  3|
|  4| An array's subsequence is a resulting sequence obtained by erasing some (possibly zero) elements from the array.
|  5|
|  6| We define that a subsequence a is more competitive than a subsequence b (of the same length) 
|  7| if in the first position where a and b differ, subsequence a has a number less than the 
|  8| corresponding number in b. For example, [1,3,4] is more competitive than [1,3,5] 
|  9| because the first position they differ is at the final number, and 4 is less than 5.
| 10|
| 11|
| 12| Example 1:
| 13|
| 14|   Input: nums = [3,5,2,6], k = 2
| 15|   Output: [2,6]
| 16|   Explanation: Among the set of every possible subsequence: {[3,5], [3,2], [3,6], [5,2], [5,6], [2,6]}, [2,6] 
| 17|   is the most competitive.
| 18|
| 19| Example 2:
| 20|
| 21|   Input: nums = [2,4,3,3,5,4,9,6], k = 4
| 22|   Output: [2,3,3,4]
| 23| """
| 24|
| 25| """
| 26| Intuition
| 27| ------------------------------------------------
| 28| Use a mono increasing stack.
| 29|
| 30| Explanation
| 31| ------------------------------------------------
| 32| Keep a mono increasing stackas result.
| 33| If current element a is smaller then the last element in the stack,
| 34| we can replace it to get a smaller sequence.
| 35|
| 36| Before we do this,
| 37| we need to check if we still have enough elements after.
| 38| After we pop the last element from stack,
| 39| we have stack.size() - 1 in the stack,
| 40| there are A.size() - i can still be pushed.
| 41| if stack.size() - 1 + A.size() - i >= k, we can pop the stack.
| 42|
| 43| Then, is the stack not full with k element,
| 44| we push A[i] into the stack.
| 45|
| 46| Finally we return stack as the result directly.
| 47| """
| 48|
| 49| # TC: O(n)
| 50| # SC: O(k)
| 51| # where, n = nums.length
| 52| def mostCompetitive(nums: List[int], k: int) -> List[int]:
| 53|     stack = []
| 54|     for i, a in enumerate(A):
| 55|         while stack and stack[-1] > a and len(stack) - 1 + len(A) - i >= k:
| 56|             stack.pop()
| 57|         if len(stack) < k:
| 58|             stack.append(a)
| 59|     return stack

#====================================================================================================================================================
#       :: Arrays ::
#       :: MONOTONOUS STACK ::
#       :: arrays/sliding_window_maximum.py ::
#       LC_239 | Sliding Window Maximum | https://leetcode.com/problems/sliding-window-maximum/ | Hard
#====================================================================================================================================================
|  1| """
|  2| You are given an array of integers nums, there is a sliding window of size k 
|  3| which is moving from the very left of the array to the very right.
|  4| You can only see the k numbers in the window. Each time the sliding window moves right by one position.
|  5|
|  6| Return the max sliding window.
|  7|
|  8|
|  9|
| 10| Example 1:
| 11|
| 12|   Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
| 13|   Output: [3,3,5,5,6,7]
| 14|   Explanation: 
| 15|   Window position                Max
| 16|   ---------------               -----
| 17|   [1  3  -1] -3  5  3  6  7       3
| 18|    1 [3  -1  -3] 5  3  6  7       3
| 19|    1  3 [-1  -3  5] 3  6  7       5
| 20|    1  3  -1 [-3  5  3] 6  7       5
| 21|    1  3  -1  -3 [5  3  6] 7       6
| 22|    1  3  -1  -3  5 [3  6  7]      7
| 23|
| 24| Example 2:
| 25|
| 26|   Input: nums = [1], k = 1
| 27|   Output: [1]
| 28|
| 29| Example 3:
| 30|
| 31|   Input: nums = [1,-1], k = 1
| 32|   Output: [1,-1]
| 33|
| 34| Example 4:
| 35|
| 36|   Input: nums = [9,11], k = 2
| 37|   Output: [11]
| 38|
| 39| Example 5:
| 40|
| 41|   Input: nums = [4,-2], k = 2
| 42|   Output: [4]
| 43| """
| 44|
| 45| """
| 46| Explanation
| 47| ------------------------------------------------
| 48| There are a big variety of different algorithms for this problem. The most difficult,
| 49| but most efficient uses idea of decreasing deque: on each moment of time we will keep 
| 50| only decreasing numbers in it.
| 51| Let us consider the following example: 
| 52| nums = [1,3,-1,-3,5,3,6,7], k = 3. 
| 53| Let us process numbers one by one: (I will print numbers, however we will keep indexes in our stack):
| 54|
| 55|   1. We put 1 into emtpy deque: [1].
| 56|   1. New element is bigger, than previous, so we remove previous element and put new one: [3].
| 57|   1. Next element is smaller than previous, put it to the end of deque: [3, -1].
| 58|   1. Similar to previous step: [3, -1, -3].
| 59|   1. Now, let us look at the first element 3, it has index 1 in our data, what does it mean? It was to far ago, and we need to delete it: so we popleft it. So, now we have [-1, -3]. Then we check that new element is bigger than the top of our deque, so we remove two elements and have [5] in the end.
| 60|   1. New element is smaller than previous, just add it to the end: [5, 3].
| 61|   1. New element is bigger, remove elements from end, until we can put it: [6].
| 62|   1. New element is bigger, remove elements from end, until we can put it: [7].
| 63|
| 64| So, once again we have the following rules:
| 65|
| 66|   1. Elements in deque are always in decreasing order.
| 67|   1. They are always elements from last sliding window of k elements.
| 68|   1. It follows from here, that biggest element in current sliding window will be the 0-th element in it.
| 69|
| 70| Complexity: time complexity is O(n), because we iterate over our elements and for each element 
| 71| it can be put inside and outside of our deque only once. Space complexity is O(k), the maximum size of our deque.
| 72| """
| 73| # TC: O(N)
| 74| # SC: O(K)
| 75| import collections
| 76| from typing import List
| 77| class Solution:
| 78|     def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
| 79|         deq, n, ans = collections.deque([0]), len(nums), []
| 80|
| 81|         for i in range (n):
| 82|             while deq and deq[0] <= i - k:
| 83|                 deq.popleft()
| 84|             while deq and nums[i] >= nums[deq[-1]] :
| 85|                 deq.pop()
| 86|             deq.append(i)
| 87|
| 88|             ans.append(nums[deq[0]])
| 89|            
| 90|         return ans[k-1:]

#====================================================================================================================================================
#       :: Arrays ::
#       :: SLIDING WINDOW ::
#       :: arrays/minimum_size_subarray_sum.py ::
#       LC_209 | Minimum Size Subarray Sum | https://leetcode.com/problems/minimum-size-subarray-sum/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given an array of positive integers nums and a positive integer target, 
|  3| return the minimal length of a contiguous subarray [nums_l, nums_l+1, ..., nums_r-1, nums_r] 
|  4| of which the sum is greater than or equal to target. If there is no such subarray, return 0 instead.
|  5|
|  6| Example 1:
|  7|
|  8|   Input: target = 7, nums = [2,3,1,2,4,3]
|  9|   Output: 2
| 10|   Explanation: The subarray [4,3] has the minimal length under the problem constraint.
| 11|
| 12| Example 2:
| 13|
| 14|   Input: target = 4, nums = [1,4,4]
| 15|   Output: 1
| 16|
| 17| Example 3:
| 18|
| 19|   Input: target = 11, nums = [1,1,1,1,1,1,1,1]
| 20|   Output: 0
| 21| """
| 22| """
| 23| Explanation
| 24| ------------------------------------------------
| 25| The result is initialized as res = n + 1.
| 26| One pass, remove the value from sum s by doing s -= A[j].
| 27| If s <= 0, it means the total sum of A[i] + ... + A[j] >= sum that we want.
| 28| Then we update the res = min(res, j - i + 1)
| 29| Finally we return the result res
| 30| """
| 31| """
| 32| Complexity
| 33| ------------------------------------------------
| 34| Time O(N)
| 35| Space O(1)
| 36| """
| 37| # TC: O(N)
| 38| # SC: O(1)
| 39| class Solution:
| 40|     def minSubArrayLen(self, target: int, nums: List[int]) -> int:
| 41|         i, res = 0, len(nums) + 1
| 42|         for j in range(len(nums)):
| 43|             s -= nums[j]
| 44|             while s <= 0:
| 45|                 res = min(res, j - i + 1)
| 46|                 s += nums[i]
| 47|                 i += 1
| 48|         return res % (len(nums) + 1)

#====================================================================================================================================================
#       :: Arrays ::
#       :: MONOTONOUS STACK ::
#       :: arrays/shortest_subarray_with_sum_at_least_k.py ::
#       LC_862 | Shortest Subarray with Sum at Least K | https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/ | Hard
#====================================================================================================================================================
|  1| """
|  2| Return the length of the shortest, non-empty, contiguous subarray of nums with sum at least k.
|  3|
|  4| If there is no non-empty subarray with sum at least k, return -1.
|  5|
|  6| Example 1:
|  7|
|  8|   Input: nums = [1], k = 1
|  9|   Output: 1
| 10|
| 11| Example 2:
| 12|
| 13|   Input: nums = [1,2], k = 4
| 14|   Output: -1
| 15|
| 16| Example 3:
| 17|
| 18|   Input: nums = [2,-1,2], k = 3
| 19|   Output: 3
| 20| """
| 21| """
| 22| Prepare
| 23| ------------------------------------------------
| 24|
| 25| "What makes this problem hard is that we have negative values.
| 26| If you haven't already done the problem with positive integers only,
| 27| I highly recommend solving it first"
| 28|
| 29| LC_209: Minimum Size Subarray Sum
| 30|
| 31| Explanation
| 32| ------------------------------------------------
| 33| Calculate prefix sum B of list A.
| 34| B[j] - B[i] represents the sum of subarray A[i] ~ A[j-1]
| 35| Deque d will keep indexes of increasing B[i].
| 36| For every B[i], we will compare B[i] - B[d[0]] with K.
| 37|
| 38| Complexity:
| 39| ------------------------------------------------
| 40| Every index will be pushed exactly once.
| 41| Every index will be popped at most once.
| 42|
| 43| Time O(N)
| 44| Space O(N)
| 45|
| 46| How to think of such solutions?
| 47| ------------------------------------------------
| 48| Basic idea, for array starting at every A[i], find the shortest one with sum at leat K.
| 49| In my solution, for B[i], find the smallest j that B[j] - B[i] >= K.
| 50| Keep this in mind for understanding two while loops.
| 51|
| 52| What is the purpose of first while loop?
| 53| ------------------------------------------------
| 54| For the current prefix sum B[i], it covers all subarray ending at A[i-1].
| 55| We want know if there is a subarray, which starts from an index, ends at A[i-1] and has at least sum K.
| 56| So we start to compare B[i] with the smallest prefix sum in our deque, which is B[D[0]], hoping that [i] - B[d[0]] >= K.
| 57| So if B[i] - B[d[0]] >= K, we can update our result res = min(res, i - d.popleft()).
| 58| The while loop helps compare one by one, until this condition isn't valid anymore.
| 59|
| 60| Why we pop left in the first while loop?
| 61| ------------------------------------------------
| 62| This the most tricky part that improve my solution to get only O(N).
| 63| D[0] exists in our deque, it means that before B[i], we didn't find a subarray whose sum at least K.
| 64| B[i] is the first prefix sum that valid this condition.
| 65| In other words, A[D[0]] ~ A[i-1] is the shortest subarray starting at A[D[0]] with sum at least K.
| 66| We have already find it for A[D[0]] and it can't be shorter, so we can drop it from our deque.
| 67|
| 68| What is the purpose of second while loop?
| 69| ------------------------------------------------
| 70| To keep B[D[i]] increasing in the deque.
| 71|
| 72| Why keep the deque increase?
| 73| ------------------------------------------------
| 74| If B[i] <= B[d.back()] and moreover we already know that i > d.back(), it means that compared with d.back(),
| 75| B[i] can help us make the subarray length shorter and sum bigger. So no need to keep d.back() in our deque.
| 76|
| 77| More detailed on this, we always add at the LAST position
| 78| B[d.back] <- B[i] <- ... <- B[future id]
| 79| B[future id] - B[d.back()] >= k && B[d.back()] >= B[i]
| 80| B[future id] - B[i] >= k too
| 81|
| 82|so no need to keep B[d.back()]
| 83|
| 84| """
| 85|
| 86| """
| 87| Detailed Explanation
| 88| ------------------------------------------------
| 89| Given B[i1], we find B[j] which is smaller than B[i1]-K, i.e. the js in the ****** part.
| 90| Among all these js, we want the max j to make the subarray shortest.
| 91|
| 92|   index j with an increasing order of B[j]
| 93| 
| 94|             B[i1] 
| 95|             V     
| 96|   *****######.....               <--- B[i2] (the new input with i2 > i1)
| 97|     ^  ^          
| 98|     |  B[i1]-K    
| 99|     |             
|100|    B[i2] - K, if B[i2] < B[i1], skip because i2 - max(***) > i1 - max(*****)
|101|               otherwise, the chance is we find a better j in the #### part
|102|
|103|   ****** part for B[j] < B[i1] - K
|104|   ...... part for B[j] > B[i1]
|105|
|106| Suppose we have found the max j in the ****** part for B[i1], and B[i2] comes in now.
|107|
|108| The key is to skip the inferior solution space that we do NOT need to search.
|109|
|110|    * For all ****** part, if B[i2] < B[i1], then B[i2] - K points to somewhere in the ****** part,
|111|      say the prefix *** are those js such that B[j] < B[i2] - K. The max of j 
|112|      in *** must be smaller than the max of j in *****, which means the distance 
|113|      from them to i2 can only be longer; Otherwise, if B[i2] >= B[i1], the target j 
|114|      would be somewhere in the ##### part. So we do not need to search ****** part.
|115|
|116|    * The js in ..... part are worse than i1, because such B[j]s are larger than B[i1] 
|117|      and further away from i2. Skip them too.
|118|
|119| So essentially, only the ##### part is worth searching for B[i2]. All the other parts are irrelavent.
|120| A deque fits our purpose, it pops the ***** and ..... part as B[i1] comes in at the first place.
|121| """
|122| # TC: O(N)
|123| # SC: O(N)
|124| import collections
|125| from typing import List
|126| class Solution:
|127|     def shortestSubarray(self, nums: List[int], k: int) -> int:
|128|         d = collections.deque([[-1, 0]])
|129|         res, cur = float('inf'), 0
|130|         for i, a in enumerate(nums):
|131|             cur += a
|132|             while d and cur - d[0][1] >= K:
|133|                 res = min(res, i - d.popleft()[0])
|134|             while d and cur <= d[-1][1]:
|135|                 d.pop()
|136|             d.append([i,  cur])
|137|         return res if res < float('inf') else -1        

#====================================================================================================================================================
#       :: Arrays ::
#       :: Prefix Sum + Sliding Window ::
#       :: arrays/shortest_subarray_with_sum_at_least_k.py ::
#       LC_862 | Shortest Subarray with Sum at Least K ( Another Way to Solve the Problem ) | https://leetcode.com/problems/shortest-subarray-with-sum-at-least-k/ | Hard
#====================================================================================================================================================
|  1| """
|  2| Return the length of the shortest, non-empty, contiguous subarray of nums with sum at least k.
|  3|
|  4| If there is no non-empty subarray with sum at least k, return -1.
|  5|
|  6| Example 1:
|  7|
|  8|   Input: nums = [1], k = 1
|  9|   Output: 1
| 10|
| 11| Example 2:
| 12|
| 13|   Input: nums = [1,2], k = 4
| 14|   Output: -1
| 15|
| 16| Example 3:
| 17|
| 18|   Input: nums = [2,-1,2], k = 3
| 19|   Output: 3
| 20| """
| 21|
| 22| """
| 23| First we build a prefix sum array psum so that psum[j]-psum[i] will be the sum of subarray A[i+1:j+1].
| 24| So once there is a psum[j] - psum[i] >= K, we have a candidate length j-i.
| 25| So we can implement slide window here to find out smallest j-i.
| 26|
| 27| To slide window, we keep append previous iterated index to a queue as left end and current iterating index as right end.
| 28| So left end would be q[0] and right end would be i. Once we found psum[i] - psum[q[0]] >= K, we have a valid window.
| 29| Then we keep reducing window's size by left popping q (increase left end) until window comes to be invalid(psum[i] - psum[q[0]] < k).
| 30|
| 31| Since K >= 1, we don't need to consider decreasing psum elements.
| 32| For example, if we have psum array = [4,8] and new psum = 2, we just retain [2] and pop out 
| 33| 8 and 4 since 2-8<1 and 2-4<1. So we can reduce our queue size by maintaining a mono-increasing queue.
| 34|
| 35| Furthermore, we can build prefix sum array and mono-increasing queue at the same time.
| 36| And we also updating our candidate subarray length simultaneously. So our queue element will be a tuple of (left_index, psum).
| 37| In such way, if current iterating element a is larger than 0, we only update slide window,
| 38| if it's smaller or equal to 0, we only update queue.
| 39|
| 40| As left and right index keeps increasing, time complexity is O(N).
| 41| """
| 42| # TC: O(N)
| 43| # SC: O(N)
| 44| import collections
| 44| def shortestSubarray(A, K):
| 45|     n = len(A)
| 46|     q, ans, psum = collections.deque([(-1,0)]), n+1, 0
| 47|     for i, a in enumerate(A):
| 48|         psum += a
| 49|         if a > 0:
| 50|             while q and psum - q[0][1] >= K:
| 51|                 ans = min(ans, i-q.popleft()[0])
| 52|         else:
| 53|             while q and psum <= q[-1][1]:
| 54|                 q.pop()
| 55|         q.append((i, psum))
| 56|     return ans if ans <= n else -1
| 57| """
| 58| Follow-up to LC-862 : Shortest Subarray with Sum at Least K ( Apple Phone Interview | Longest Subarray Sum At Most K | https://leetcode.com/discuss/interview-question/758045/ )
| 59| ------------------------------------------------
| 60| Given an array of integers (contains both positive and negative numbers),
| 61| find the longest subarray with sum at most k. Is this doable in O(n)?
| 62| """
| 63| # TC: O(N)
| 64| # SC: O(N)
| 65| """
| 66| The difficulty in this problem is that there are positive and negative numbers.
| 67| Here's a motivating example to use the solution for positive numbers and extend it to negative numbers.
| 68|
| 69| For just positive numbers, we start with a [i=0, j=0] window. We need to keep a 
| 70| running window sum (ie., prefix sum) and slowly expand j.
| 71| Whenever the running sum exceeds k we compress the window from the left and increment i.
| 72| We know we can do this because given only positive numbers, all subsequent window sums 
| 73| will be larger than the current one and no valid answer is possible if i remained fixed.
| 74| (In the solution below, I'm also tracking the window indices)
| 75| """
| 77| def longest_subarray_with_at_most_k(A, k):
| 78|     n = len(A)
| 79|     res, si, sj = 0, None, None
| 80|
| 81|     ws = 0  # window sum
| 82|     i = 0
| 83|     for j in range(n):
| 84|         ws += A[j]
| 85|         while i < j and ws > k:
| 86|             ws -= A[i]
| 87|             i += 1
| 88|
| 89|         if (j - i + 1) > res:
| 90|             res = (j - i + 1)
| 91|             si, sj = i, j
| 92|
| 93|     return (res, A[si:sj+1]) if res != 0 else (res, [])
| 94| """
| 94| With negative numbers, the problem is that the running window sum 
| 95| is insufficient to determine if we can stop at some j or not. It's obviously possible that 
| 96| there is a huge negative number after the current j that will reduce the sum to be below k and have a larger window.
| 97|
| 98| To account for this, we will track just one more thing - For each index j,
| 99| we will check if the remaining part of the array (ie., A[j+1:]) is negative or not.
|100| That way, we just need to update the check from ws > k to ws + neg_sum[j] > k.
|101| That simply means that (a) the current window sum exceeds k and (b) the 
|102| remainder of the array cannot possibly decrease the value any more.
|103| Note that for the all positives case, neg_sum[j] = 0 for all j so it directly reduces to the first version.
|104| """
|105| def longest_subarray_with_at_most_k(A, k):
|106|     n = len(A)
|107|     res, si, sj = 0, None, None
|108|    
|109|     neg_sum = [0] * n   # ADDED
|110|     rs = 0
|111|     for j in range(n-1, -1, -1):
|112|         neg_sum[j] = rs
|113|         rs = min(0, rs + A[j])
|114|
|115|     ws = 0
|116|     i = 0
|117|     for j in range(n):
|118|         ws += A[j]
|119|         while i < j and ws + neg_sum[j] > k:  # CHANGED
|120|             ws -= A[i]
|121|             i += 1
|122|       
|123|         if (j - i + 1) > res:
|124|             res = (j - i + 1)
|125|             si, sj = i, j
|126|     return (res, A[si:sj+1]) if res != 0 else (res, [])
|127| """
|128| You can verify it with this:
|129| """
|130|
|131| for A, k in [
|132|     ([9,1,2,3,4,5], 7),
|133|     ([5, -10, 7, -20, 57], -22),
|134|     ([-5, 8, -14, 2, 4, 12], 5),
|135|     ([1, 2, 1, 0, 1, -8, -9, 0], 4)
|136| ]:
|137|     print(A, k, '=>', longest_subarray_with_at_most_k(A, k))
|138|
|139| #[9, 1, 2, 3, 4, 5] 7 => (3, [1, 2, 3])
|140| #[5, -10, 7, -20, 57] -22 => (3, [-10, 7, -20])
|141| #[-5, 8, -14, 2, 4, 12] 5 => (5, [-5, 8, -14, 2, 4])
|142| #[1, 2, 1, 0, 1, -8, -9, 0] 4 => (8, [1, 2, 1, 0, 1, -8, -9, 0])

#====================================================================================================================================================
#       :: Trees ::
#       :: DYNAMIC PROGRAMMING ::
#       :: GREEDY ::
#       :: arrays/minimum_cost_tree_from_leaf_values.py ::
#       LC_1130 | Minimum Cost Tree From Leaf Values ( How not to solve the problem )| https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given an array arr of positive integers, consider all binary trees such that:
|  3|
|  4|   * Each node has either 0 or 2 children;
|  5|   * The values of arr correspond to the values of each leaf in an in-order traversal of the tree.
|  6|     (Recall that a node is a leaf if and only if it has 0 children.)
|  7|   * The value of each non-leaf node is equal to the product of the largest leaf value 
|  8|     in its left and right subtree respectively.
|  9|
| 10| Among all possible binary trees considered, return the smallest possible sum of the values of each non-leaf node.
| 11| It is guaranteed this sum fits into a 32-bit integer. 
| 12|
| 13| Example 1:
| 14|
| 15|   Input: arr = [6,2,4]
| 16|   Output: 32
| 17|   Explanation:
| 18|   There are two possible trees.  The first has non-leaf node sum 36, and the second has non-leaf node sum 32.
| 19|
| 20|       24            24
| 21|      /  \          /  \
| 22|     12   4        6    8
| 23|    /  \               / \
| 24|   6    2             2   4
| 25| """
| 26|
| 27| """
| 28| DP Solution
| 30| ------------------------------------------------
| 31| Find the cost for the interval [i,j].
| 32| To build up the interval [i,j],
| 33| we need to split it into left subtree and sub tree,
| 34| dp[i, j] = dp[i, k] + dp[k + 1, j] + max(A[i, k]) * max(A[k + 1, j])
| 35|
| 36| If you don't understand dp solution,
| 37| I won't explain it more and you won't find the answer here.
| 38| Take your time,
| 39| read any other solutions ( like : https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/discuss/478708/RZ-Summary-of-all-the-solutions-I-have-learned-from-Discuss-in-Python ),
| 40| and come back at your own will.
| 41|
| 42| If you got it, continue to read.
| 43|
| 44| DP Complexity
| 45| Second question after this dp solution,
| 46| what's the complexity?
| 47| N^2 states and O(N) to find each.
| 48| So this solution is O(N^3) time and O(N^2) space.
| 49|
| 50| You thought it's fine.
| 51| After several nested for loop, you got a happy green accepted.
| 52| You smiled and released a sigh as a winner.
| 53|
| 54| What a great practice for DP skill!
| 55| Then you noticed it's medium.
| 56| That's it, just a standard medium problem of dp.
| 57| Nothing can stop you. Even dp problem.
| 58| """
| 59|
| 60| # 1) Dynamic programming approach ---> O(n ^ 3)
| 61| #------------------------------------------------
| 61| # We are given a list of all the leaf nodes values for certain binary trees,
| 62| # but we do not know which leaf nodes belong to left subtree and which leaf nodes belong to right subtree.
| 63| # Since the given leaf nodes are result of inorder traversal, we know there will be pivots 
| 64| # that divide arr into left and right, nodes in the left build left subtree and nodes 
| 65| # in the right build right subtree. For each subtree, if we know the minimum sum, 
| 66| # we can use it to build the result of the parent tree, so the problem can be divided into subproblems,
| 66| # and we have the following general transition equation (res(i, j) 
| 67| # means the minimum non-leaf nodes sum with leaf nodes from arr[i] to arr[j]):
| 68|
| 69| #     for k from i to j
| 70| #         res(i, j) = min(res(i, k) + res(k + 1, j) + max(arr[i] ... arr[k]) * max(arr[k + 1] ... arr[j]))
| 71|
| 72| # TC: O(N^3); SC: O(N^2)
| 73| # Top down code with memorization ---> O(n ^ 3)
| 74| class Solution:
| 75|     def mctFromLeafValues(self, arr: List[int]) -> int:
| 76|         return self.helper(arr, 0, len(arr) - 1, {})
| 77|
| 78|     def helper(self, arr, l, r, cache):
| 79|         if (l, r) in cache:
| 80|             return cache[(l, r)]
| 81|         if l >= r:
| 82|             return 0
| 83|
| 84|         res = float('inf')
| 85|         for i in range(l, r):
| 86|             rootVal = max(arr[l:i+1]) * max(arr[i+1:r+1])
| 87|             res = min(res, rootVal + self.helper(arr, l, i, cache) + self.helper(arr, i + 1, r, cache))
| 88|
| 89|         cache[(l, r)] = res
| 90|         return res
| 91|
| 92| # TC: O(N^3); SC: O(N^2)
| 93| # Bottom up code ---> O(n ^ 3)
| 94| class Solution:
| 95|     def mctFromLeafValues(self, arr: List[int]) -> int:
| 96|         n = len(arr)
| 97|         dp = [[float('inf') for _ in range(n)] for _ in range(n)]
| 98|         for i in range(n):
| 99|             dp[i][i] = 0
|100|
|101|         for l in range(2, n + 1):
|102|             for i in range(n - l + 1):
|103|                 j = i + l - 1
|104|                 for k in range(i, j):
|105|                     rootVal = max(arr[i:k+1]) * max(arr[k+1:j+1])
|106|                     dp[i][j] = min(dp[i][j], rootVal + dp[i][k] + dp[k + 1][j])
|107|         return dp[0][n - 1]
|108|
|109| # 2) Greedy approach ---> O(n ^ 2)
|110| #------------------------------------------------
|111| # Above approach is kind of like brute force since we calculate and compare the results all possible pivots.
|112| # To achieve a better time complexity, one important observation is that when we build each level of the binary tree,
|113| # it is the max left leaf node and max right lead node that are being used,
|114| # so we would like to put big leaf nodes close to the root. Otherwise, taking the leaf node with max value 
|115| # in the array as an example, if its level is deep, for each level above it, 
|116| # its value will be used to calculate the non-leaf node value, which will result in a big total sum.
|117|
|118| # With above observation, the greedy approach is to find the smallest value in the array,
|119| # use it and its smaller neighbor to build a non-leaf node, then we can safely delete 
|120| # it from the array since it has a smaller value than its neightbor so it will never be used again.
|121| # Repeat this process until there is only one node left in the array (which means we cannot build a new level any more)
|122|
|123| # TC: O(N^3); SC: O(N^2)
|124| # Greedy approach ---> O(n ^ 2)
|125| class Solution:
|126|     def mctFromLeafValues(self, arr: List[int]) -> int:
|127|         res = 0
|128|         while len(arr) > 1:
|129|             index = arr.index(min(arr))
|130|             if 0 < index < len(arr) - 1:
|131|                 res += arr[index] * min(arr[index - 1], arr[index + 1])
|132|             else:
|133|                 res += arr[index] * (arr[index + 1] if index == 0 else arr[index - 1])
|134|             arr.pop(index)
|135|         return res

#====================================================================================================================================================
#       :: Trees ::
#       :: MONOTONOUS STACK ::
#       :: arrays/minimum_cost_tree_from_leaf_values.py ::
#       LC_1130 | Minimum Cost Tree From Leaf Values ( The correct way to solve the problem )| https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given an array arr of positive integers, consider all binary trees such that:
|  3|
|  4|   * Each node has either 0 or 2 children;
|  5|   * The values of arr correspond to the values of each leaf in an in-order traversal of the tree.
|  6|     (Recall that a node is a leaf if and only if it has 0 children.)
|  7|   * The value of each non-leaf node is equal to the product of the largest leaf value 
|  8|     in its left and right subtree respectively.
|  9|
| 10| Among all possible binary trees considered, return the smallest possible sum of the values of each non-leaf node.
| 11| It is guaranteed this sum fits into a 32-bit integer. 
| 12|
| 13| Example 1:
| 14|
| 15|   Input: arr = [6,2,4]
| 16|   Output: 32
| 17|   Explanation:
| 18|   There are two possible trees.  The first has non-leaf node sum 36, and the second has non-leaf node sum 32.
| 19|
| 20|       24            24
| 21|      /  \          /  \
| 22|     12   4        6    8
| 23|    /  \               / \
| 24|   6    2             2   4
| 25| """
| 26|
| 27| """
| 28| True story
| 29| ------------------------------------------------
| 30| So you didn't Read and Upvote this post [ https://leetcode.com/problems/minimum-cost-tree-from-leaf-values/discuss/339959/One-Pass-O(N)-Time-and-Space ].
| 31| (upvote is a good mark of having read)
| 32| One day, you meet exactly the same solution during an interview.
| 33| Your heart welled over with joy,
| 34| and you bring up your solution with confidence.
| 35|
| 36| Happened at a Google interview ... so this is no joke.
| 37|
| 38| One week later, you receive an email.
| 39| The second paragraph starts with a key word "Unfortunately".
| 40|
| 41| What the heck!?
| 42| ------------------------------------------------
| 43| You solved the interview problem perfectly,
| 44| but the company didn't appreciate your talent.
| 45| What's more on earth did they want?
| 46| WHY?
| 47|
| 48| Why
| 49| ------------------------------------------------
| 50| Here is the reason.
| 51| This is not a dp problem at all.
| 52|
| 53| Because dp solution test all ways to build up the tree,
| 54| including many unnecessay tries.
| 55| Honestly speaking, it's kinda of brute force.
| 56| Yes, brute force testing, with memorization.
| 57|
| 58| Intuition
| 59| ------------------------------------------------
| 60| Let's review the problem again.
| 61|
| 62| When we build a node in the tree, we compared the two numbers a and b.
| 63| In this process,
| 64| the smaller one is removed and we won't use it anymore,
| 65| and the bigger one actually stays.
| 66|
| 67| The problem can translated as follows:
| 68| Given an array A, choose two neighbors in the array a and b,
| 69| we can remove the smaller one min(a,b) and the cost is a * b.
| 70| What is the minimum cost to remove the whole array until only one left?
| 71|
| 72| To remove a number a, it needs a cost a * b, where b >= a.
| 73| So a has to be removed by a bigger number.
| 74| We want minimize this cost, so we need to minimize b.
| 75|
| 76| b has two candidates, the first bigger number on the left,
| 77| the first bigger number on the right.
| 78|
| 79| The cost to remove a is a * min(left, right).
| 80|
| 81| Solution 1
| 82| ------------------------------------------------
| 83| With the intuition above in mind,
| 84| the explanation is short to go.
| 85|
| 86| We remove the element form the smallest to bigger.
| 87| We check the min(left, right),
| 88| For each element a, cost = min(left, right) * a
| 89|
| 90| Time O(N^2)
| 91| Space O(N)
| 92| """
| 93| def mctFromLeafValues(self, A):
| 94|     res = 0
| 95|     while len(A) > 1:
| 96|         i = A.index(min(A))
| 97|         res += min(A[i - 1:i] + A[i + 1:i + 2]) * A.pop(i)
| 98|     return res
| 99| """
|100| Solution 2: Stack Soluton
|101| ------------------------------------------------
|102| We decompose a hard problem into reasonable easy one:
|103| Just find the next greater element in the array, on the left and one right.
|104| Refer to the problem 503. Next Greater Element II
|105|
|106| Time: O(N) for one pass
|107| Space: O(N) for stack in the worst cases
|108| """
|109| def mctFromLeafValues(self, A):
|110|     res = 0
|111|     stack = [float('inf')]
|112|     for a in A:
|113|         while stack[-1] <= a:
|114|             mid = stack.pop()
|115|             res += mid * min(stack[-1], a)
|116|         stack.append(a)
|117|     while len(stack) > 2:
|118|         res += stack.pop() * stack[-1]
|119|     return res

#====================================================================================================================================================
#       :: Arrays ::
#       :: MONOTONOUS STACK ::
#       :: arrays/constrained_subsequence_sum.py ::
#       LC_1425 | Constrained Subsequence Sum | https://leetcode.com/problems/constrained-subsequence-sum/ | Hard
#====================================================================================================================================================
|  1| """
|  2| Given an integer array nums and an integer k, return the maximum sum of a non-empty subsequence of that array 
|  3| such that for every two consecutive integers in the subsequence, nums[i] and nums[j], where i < j, 
|  4| the condition j - i <= k is satisfied.
|  5|
|  6| A subsequence of an array is obtained by deleting some number of elements (can be zero) from the array, 
|  7| leaving the remaining elements in their original order.
|  8|
|  9| Example 1:
| 10|
| 11|   Input: nums = [10,2,-10,5,20], k = 2
| 12|   Output: 37
| 13|   Explanation: The subsequence is [10, 2, 5, 20].
| 14|
| 15| Example 2:
| 16|
| 17|   Input: nums = [-1,-2,-3], k = 1
| 18|   Output: -1
| 19|   Explanation: The subsequence must be non-empty, so we choose the largest number.
| 20|
| 21| Example 3:
| 22|
| 23|   Input: nums = [10,-2,-10,-5,20], k = 2
| 24|   Output: 23
| 25|   Explanation: The subsequence is [10, -2, -5, 20].
| 26| """
| 27|
| 28| """
| 29| Intuition
| 30| ------------------------------------------------
| 31| We need to know the maximum in the window of size k.
| 32| ------------------------------------------------
| 33| * Use heap will be O(NlogN)
| 34| * Use TreeMap will be O(NlogK)
| 35| * Use deque will be O(N)
| 36|
| 30| Explanation
| 31| ------------------------------------------------
| 32| We scan the array from 0 to n-1, keep "promising" elements in the deque. 
| 33| The algorithm is amortized O(n) as each element is put and polled once.
| 34|
| 35| At each i, we keep "promising" elements, which are potentially max number in window [i-(k-1),i] 
| 36| or any subsequent window. This means :
| 37|
| 38|   1. If an element in the deque and it is out of i-(k-1), we discard them. We just need to poll from the head,
| 39|      as we are using a deque and elements are ordered as the sequence in the array
| 40|
| 41|   2. Now only those elements within [i-(k-1),i] are in the deque. We then discard elements smaller 
| 42|      than a[i] from the tail. This is because if a[x] <a[i] and x<i, then a[x] 
| 43|      has no chance to be the "max" in [i-(k-1),i], or any other subsequent window: a[i] 
| 44|      would always be a better candidate.
| 45|
| 46|   3. As a result elements in the deque are ordered in both sequence in array and their value.
| 47|      At each step the head of the deque is the max element in [i-(k-1),i].
| 48|
| 49| Algorithm
| 50| ------------------------------------------------
| 51| Update res[i],
| 52| where res[i] means the maximum result you can get if the last element is A[i].
| 53|
| 54| I directly modify on the input A,
| 55| if you don't like it,
| 56| use a copy of A
| 57|
| 58| Keep a decreasing deque q,
| 59| deque[0] is the maximum result in the last element of result.
| 60|
| 61| If deque[0] > 0. we add it to A[i]
| 62|
| 63| In the end, we return the maximum res.
| 64| """
| 65| """
| 66| Complexity
| 67| ------------------------------------------------
| 68| Because all element are pushed and popped at most once.
| 69| Time O(N)
| 70|
| 71| Because at most O(K) elements in the deque.
| 72| Space O(K)
| 73| """
| 74| """
| 75| To get yourself familiarized - solve the following problems:
| 76| ------------------------------------------------
| 77| 239 - Sliding Window Maximum
| 78| 862 - Shortest Subarray with Sum at Least K
| 78| """
| 79| # TC: O(N)
| 70| # SC: O(K)
| 80| import collections
| 81| from typing import List
| 82| class Solution:
| 83|     def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
| 84|         """
| 85|         Transition: nums[i] = max(0, nums[i - k], nums[i - k + 1], .., nums[i - 1]) + nums[i]. (@lee215 modifies the input nums directly)            
| 86|         Translated into a traditional dp: dp[i] = max(0, dp[i - k], dp[i - k + 1], .., dp[i -1]) + nums[i]
| 87|         dp[i] is the max sum we can have from nums[:i] when nums[i] has been chosen.
| 88|         """ 
| 89|         # `deque` stores dp[i - k], dp[i-k+1], .., dp[i - 1] whose values are larger than 0 in a decreasing order
| 90|         # Note that the length of `deque` is not necessarily `k`. The values smaller than dp[i-1] will be discarded. If u r confused, go on and come back later. 
| 91|         deque = collections.deque() 
| 92|         for i in range(len(nums)):
| 93|             # deque[0] is the max of (0, dp[i - k], dp[i-k+1], .., dp[i - 1])
| 94|             nums[i] += deque[0] if deque else 0 
| 95|             # 1. We always want to retrieve the max of (0, dp[i - k], dp[i-k+1], .., dp[i - 1]) from `deque`
| 96|             # 2. We expect dp[i] to be added to `deque` so that we can compute dp[i + 1] in the next iteration
| 97|             # 3. So, if dp[i] is larger than some old values, we can discard them safely.
| 98|             # 4. As a result, the length of `deque` is not necessarily `k`
| 99|             while len(deque) and nums[i] >= deque[-1]:
|100|                 deque.pop()
|101|             # no need to store the negative value
|102|             if nums[i] > 0:
|103|                 deque.append(nums[i])
|104|             # we do not need the value of nums[i - k] when computing dp[i+1] in the next iteration, because `j - i <= k` has to be satisfied.
|105|             if i >= k and deque and deque[0] == nums[i - k]:
|106|                 deque.popleft()
|107|         return max(nums)
|108| """
|109| Follow-up Solution (If we are asked to not modify the input array in an interview).
|110| In this case, the solutions above would require O(n) space.
|111| Instead of copying the input array, we can maintain another deque of size k, 
|112| reducing the space complexity to O(k).
|113| """
|114| # TC: O(N)
|115| # SC: O(K)
|116| import collections
|117| from typing import List
|118| class Solution:
|119| def constrainedSubsetSum(self, nums: List[int], k: int) -> int:
|120|     window = deque()
|121|     max_queue = deque()
|122|     max_sum = float('-inf')
|123|
|124|     for i in range(len(nums)):
|125|         window.append(nums[i])
|126|         if max_queue:
|127|             window[-1] += max_queue[0]
|128|
|129|         max_sum = max(max_sum, window[-1])
|130|
|131|         while max_queue and window[-1] > max_queue[-1]:
|132|             max_queue.pop()
|133|         if window[-1] > 0:
|134|             max_queue.append(window[-1])
|135|         if i >= k:
|136|             if max_queue and window[0] == max_queue[0]:
|137|                 max_queue.popleft()
|138|             window.popleft()
|139|
|140|     return max_sum
    

#====================================================================================================================================================
#       :: Trees ::
#       :: Prefix Sum and DFS ::
#       :: trees/path_sum_iii.py ::
#       LC-437 | Path Sum III | https://leetcode.com/problems/path-sum-iii/ | Medium
#====================================================================================================================================================
|  1| """
|  2| Given the root of a binary tree and an integer targetSum, return the number of paths where the sum of the values 
|  3| along the path equals targetSum.
|  4|
|  5| The path does not need to start or end at the root or a leaf, but it must go downwards 
|  6| (i.e., traveling only from parent nodes to child nodes).
|  7|
|  8| Example 1:
|  9|   
| 10|                ( 10 )
| 11|        *******/*     \
| 12|         *+---/  *  +++\++++++
| 13|         /*( 5 )\ *  + ( -3 ) +
| 14|        /  */  \/  *  +  \     +
| 15|       /   /*  /\   *  +  \     +
| 16|      / ( 3 )*/( 2 ) *  + ( 11 ) +
| 17|      \__/_\_/*    \  *  ++++++++++
| 18|        /   \  *    \  *
| 19|     ( 3 )( -2 )* ( 1 ) *
| 20|                 *********
| 21|
| 22|   Input: root = [10,5,-3,3,2,null,11,3,-2,null,1], targetSum = 8
| 23|   Output: 3
| 24|   Explanation: The paths that sum to 8 are shown.
| 25|
| 26| Example 2:
| 27|   Input: root = [5,4,8,11,null,13,4,7,2,null,null,5,1], targetSum = 22
| 28|   Output: 3
| 29|                
| 30| """
| 31|
| 32| """
| 33| If we can have any solution, that this problem is indeed medium.
| 34| However if we want to have O(n) time solution, in my opinion it is more like hard, 
| 35| because we need to use dfs + hash table in the same time.
| 36|
| 37| First of all, let us evaluate cumulative sum of our tree, what does it mean?
| 38| Here is an example: for each node we evaluate sum of path from this node to root.
| 39|              (10)                               (10)
| 40|            /      \                           /      \
| 41|         ( 5 )     ( -3 )                  ( 15 )      ( 7 )
| 42|         /   \        \                    /    \          \  
| 43|        /     \        \                  /      \          \
| 44|     ( 3 )   ( 2 )    ( 11 )           ( 18 )   ( 17 )     ( 18 )
| 45|     /   \      \                      /    \      \
| 46|  ( 3 ) ( -2 ) ( 1 )                ( 21 ) ( 16 ) ( 18 )
| 47| Also we evaluate number of nodes in our tree. What we need to find now?
| 48| Number of pairs of two nodes, one of them is descendant of another 
| 49| and difference of cumulative sums is equal to sum. Let me explain my dfs(self, root, sum) function:
| 50|
| 51| If we reached None, just go back.
| 52| We have self.count hash table, where we put root.val + sum: number we are looking for when we visited this node: for example if we are in node with value 15 now (see our example), then we want to find node with values 15+8 inside.
| 53| Also, we add to our final result number of solution for our root.val.
| 54| Run recursively our dfs for left and right children
| 55| Remove our root.val + sum from our hash table, because we are not anymore in root node.
| 56| Now, let us consider our pathSum(self, root, sum) function:
| 57|
| 58| If tree is empty, we return 0
| 59| Define some global variables.
| 60| Run our cumSum function to evaluate cumulative sums
| 61| Run dfs method.
| 62| In the end we need to subtract self.n*(sum == 0) from our result. Why? Because if sum = 0, then we will count exactly self.n empty cumulative sums, which consist of zero elements.
| 63|
| 64| Complexity: Time complexity to evaluate cumulative sums is O(n). The same time complexity is for dfs function. Space complexity is also O(n), because we need to keep this amount of information in our count hash table.
| 65|
| 66| NOTE: We do not really need to evaluate cumlative sums, we can do in on the fly.
| 67|       Also if we do it on the fly, we do not need to check case if sum == 0.
| 68|
| 69| """
| 70|
| 71| # Prefix Sum + DFS Approach
| 72| # TC: O(N)
| 73| # SC: O(N)
| 74| # Definition for a binary tree node.
| 75| import collections
| 76| class TreeNode:
| 77|     def __init__(self, val=0, left=None, right=None):
| 78|         self.val = val
| 79|         self.left = left
| 80|         self.right = right
| 81|
| 82| class Solution:
| 83|     def dfs(self, root, sum, root_sum):
| 84|         if not root: return None
| 85| 
| 86|         root_sum += root.val
| 87|         self.result += self.count[root_sum]    # process result before adding to the counts -> no need to count/adjust
| 88|         self.count[root_sum + sum] += 1
| 89|         self.dfs(root.left, sum, root_sum)
| 90|         self.dfs(root.right, sum, root_sum)
| 91|         self.count[root_sum + sum] -= 1
| 92|
| 93|     def pathSum(self, root: TreeNode, targetSum: int) -> int:
| 94|         self.result, self.count = 0, collections.defaultdict(int)
| 95|         self.count[sum] = 1
| 96|         self.dfs(root, sum, 0)
| 97|         return self.result


#====================================================================================================================================================
#       :: Arrays :: 
#       :: arrays/find_the_duplicate_number.py ::
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
#       :: arrays/remove_duplicates_from_unsorted_array.py ::
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
| 28| # Soluton-2: If we need to preserve the input arrays/list
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
#       :: arrays/remove_duplicates_from_sorted_array.py ::
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

