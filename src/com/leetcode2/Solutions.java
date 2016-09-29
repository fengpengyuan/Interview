package com.leetcode2;

import java.io.IOException;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.StringTokenizer;
import java.util.TreeMap;
import java.util.Vector;

public class Solutions {

	public ListNode mergeKLists(ListNode[] lists) {
		if (lists.length == 0)
			return null;
		return mergeKLists(lists, 0, lists.length - 1);
	}

	public ListNode mergeKLists(ListNode[] lists, int left, int right) {
		if (left >= right)
			return lists[left];
		int mid = (left + right) / 2;
		ListNode l = mergeKLists(lists, left, mid);
		ListNode r = mergeKLists(lists, mid + 1, right);
		return mergeTwoLists(l, r);
	}

	public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		if (l1 == null || l2 == null)
			return l1 == null ? l2 : l1;
		ListNode dummy = new ListNode(0);
		ListNode head = dummy;
		while (l1 != null && l2 != null) {
			if (l1.val < l2.val) {
				head.next = l1;
				l1 = l1.next;
			} else {
				head.next = l2;
				l2 = l2.next;
			}
			head = head.next;
		}
		if (l1 != null)
			head.next = l1;
		if (l2 != null)
			head.next = l2;
		return dummy.next;
	}

	public ListNode mergeKLists2(ListNode[] lists) {
		if (lists.length == 0)
			return null;
		class ListNodeComparator implements Comparator<ListNode> {
			@Override
			public int compare(ListNode l1, ListNode l2) {
				return l1.val - l2.val;
			}
		}

		PriorityQueue<ListNode> heap = new PriorityQueue<ListNode>(
				lists.length, new ListNodeComparator());
		ListNode dummy = new ListNode(0);
		ListNode head = dummy;
		for (int i = 0; i < lists.length; i++) {
			if (lists[i] != null)
				heap.offer(lists[i]);
		}
		while (!heap.isEmpty()) {
			ListNode top = heap.poll();
			head.next = top;
			head = head.next;
			if (top.next != null)
				heap.offer(top.next);
		}
		return dummy.next;
	}

	public int removeDuplicates(int[] nums) {
		if (nums.length < 2)
			return nums.length;
		int j = 1;
		for (int i = 1; i < nums.length; i++) {
			if (nums[i] != nums[i - 1]) {
				nums[j++] = nums[i];
			}
		}
		return j;
	}

	public int removeDuplicates2(int[] nums) {
		if (nums.length < 3)
			return nums.length;
		int j = 1;
		int count = 1;
		for (int i = 1; i < nums.length; i++) {
			if (nums[i] != nums[i - 1]) {
				nums[j++] = nums[i];
				count = 1;
			} else {
				count++;
				if (count == 2) {
					nums[j++] = nums[i];
				}
			}
		}
		return j;
	}

	public int removeDuplicates3(int[] nums) {
		if (nums.length < 3)
			return nums.length;
		int j = 2;
		for (int i = 2; i < nums.length; i++) {
			if (nums[i] != nums[j - 1] || nums[i] != nums[j - 2])
				nums[j++] = nums[i];
		}
		return j;
	}

	public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
		if (intervals.size() == 0) {
			intervals.add(newInterval);
			return intervals;
		}
		List<Interval> res = new ArrayList<Interval>();
		// Collections.sort(intervals, new Comparator<Interval>(){
		// @Override
		// public int compare(Interval i1, Interval i2){
		// return i1.start-i2.start;
		// }
		// });
		boolean inserted = false;
		for (int i = 0; i < intervals.size(); i++) {
			Interval interval = intervals.get(i);
			if (interval.start < newInterval.start) {
				insertInterval(res, interval);
			} else {
				insertInterval(res, newInterval);
				inserted = true;
				insertInterval(res, interval);
			}
		}
		if (!inserted) {
			insertInterval(res, newInterval);
		}
		return res;
	}

	public void insertInterval(List<Interval> res, Interval newInterval) {
		if (res.size() == 0) {
			res.add(newInterval);
			return;
		}
		Interval last = res.get(res.size() - 1);
		if (last.end < newInterval.start) {
			res.add(newInterval);
		} else {
			last.end = Math.max(last.end, newInterval.end);
		}
	}

	public List<Interval> insert2(List<Interval> intervals, Interval newInterval) {
		List<Interval> res = new ArrayList<Interval>();
		boolean inserted = false;
		for (Interval interval : intervals) {
			if (newInterval.start < interval.start && !inserted) {
				inserted = true;
				insertInterval(res, newInterval);
			}
			insertInterval(res, interval);
		}
		if (!inserted)
			insertInterval(res, newInterval);
		return res;
	}

	public List<Interval> merge(List<Interval> intervals) {
		if (intervals.size() < 2)
			return intervals;
		Collections.sort(intervals, new Comparator<Interval>() {
			@Override
			public int compare(Interval i1, Interval i2) {
				return i1.start - i2.start;
			}
		});
		List<Interval> res = new ArrayList<Interval>();
		res.add(intervals.get(0));
		for (int i = 1; i < intervals.size(); i++) {
			Interval interval = intervals.get(i);
			Interval last = res.get(res.size() - 1);
			if (last.end < interval.start) {
				res.add(interval);
			} else {
				last.end = Math.max(last.end, interval.end);
			}
		}
		return res;
	}

	public List<List<String>> groupAnagrams(String[] strs) {
		List<List<String>> res = new ArrayList<List<String>>();
		Map<String, List<String>> map = new HashMap<String, List<String>>();
		for (String str : strs) {
			char[] chars = str.toCharArray();
			Arrays.sort(chars);
			String s = new String(chars);
			if (map.containsKey(s)) {
				map.get(s).add(str);
			} else {
				List<String> lst = new ArrayList<String>();
				lst.add(str);
				map.put(s, lst);
			}
		}
		Iterator<List<String>> it = map.values().iterator();
		while (it.hasNext()) {
			List<String> values = it.next();
			Collections.sort(values);
			res.add(values);
		}
		return res;
	}

	public List<List<Integer>> combinationSum(int[] candidates, int target) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		Arrays.sort(candidates);
		combinationSumUtil(candidates, target, sol, res, 0, 0);
		return res;
	}

	public void combinationSumUtil(int[] candidates, int target,
			List<Integer> sol, List<List<Integer>> res, int cursum, int cur) {
		if (cursum > target)
			return;
		if (cursum == target) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}
		for (int i = cur; i < candidates.length; i++) {
			cursum += candidates[i];
			sol.add(candidates[i]);
			combinationSumUtil(candidates, target, sol, res, cursum, i);
			cursum -= candidates[i];
			sol.remove(sol.size() - 1);
		}
	}

	public List<List<Integer>> combinationSum3(int k, int n) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		combinationSum3Util(0, k, n, 0, sol, res, 1);
		return res;
	}

	public void combinationSum3Util(int dep, int k, int n, int cursum,
			List<Integer> sol, List<List<Integer>> res, int cur) {
		if (dep > k || cursum > n)
			return;
		if (dep == k && cursum == n) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}
		for (int i = cur; i <= 9; i++) {
			cursum += i;
			sol.add(i);
			combinationSum3Util(dep + 1, k, n, cursum, sol, res, i + 1);
			cursum -= i;
			sol.remove(sol.size() - 1);
		}
	}

	public List<List<Integer>> combinationSum2(int[] candidates, int target) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		Arrays.sort(candidates);
		boolean[] used = new boolean[candidates.length];
		List<Integer> sol = new ArrayList<Integer>();
		combinationSum2(candidates, target, sol, res, used, 0, 0);
		return res;
	}

	public void combinationSum2(int[] candidates, int target,
			List<Integer> sol, List<List<Integer>> res, boolean[] used,
			int cur, int cursum) {
		if (cur == candidates.length || cursum > target) {
			return;
		}
		if (cursum == target) {
			res.add(new ArrayList<Integer>(sol));
			return;
		}

		for (int i = cur; i < candidates.length; i++) {
			if (!used[i]) {
				if (i != 0 && candidates[i] == candidates[i - 1]
						&& !used[i - 1])
					continue;
				used[i] = true;
				cursum += candidates[i];
				sol.add(candidates[i]);
				combinationSum2(candidates, target, sol, res, used, i, cursum);
				cursum -= candidates[i];
				sol.remove(sol.size() - 1);
				used[i] = false;
			}
		}
	}

	public void setZeroes(int[][] matrix) {
		if (matrix.length == 0)
			return;
		int m = matrix.length;
		int n = matrix[0].length;
		boolean[] row = new boolean[m];
		boolean[] col = new boolean[n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (matrix[i][j] == 0) {
					row[i] = true;
					col[j] = true;
				}
			}
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (row[i] || col[j]) {
					matrix[i][j] = 0;
				}
			}
		}
	}

	public void setZeroes2(int[][] matrix) {
		if (matrix.length == 0)
			return;
		int m = matrix.length;
		int n = matrix[0].length;

		boolean fr = false, fc = false;

		for (int i = 0; i < m; i++) {
			if (matrix[i][0] == 0) {
				fc = true;
				break;
			}
		}

		for (int i = 0; i < n; i++) {
			if (matrix[0][i] == 0) {
				fr = true;
				break;
			}
		}

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				if (matrix[i][j] == 0) {
					matrix[i][0] = 0;
					matrix[0][j] = 0;
				}
			}
		}

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				if (matrix[i][0] == 0 || matrix[0][j] == 0) {
					matrix[i][j] = 0;
				}
			}
		}

		if (fr) {
			for (int i = 0; i < n; i++) {
				matrix[0][i] = 0;
			}
		}
		if (fc) {
			for (int i = 0; i < m; i++) {
				matrix[i][0] = 0;
			}
		}
	}

	public boolean searchMatrix(int[][] matrix, int target) {
		if (matrix.length == 0)
			return false;
		int i = 0;
		int j = matrix[0].length - 1;

		while (i < matrix.length && j >= 0) {
			if (matrix[i][j] == target)
				return true;
			else if (matrix[i][j] > target)
				j--;
			else
				i++;
		}
		return false;
	}

	public boolean searchMatrix2(int[][] matrix, int target) {
		if (matrix.length == 0)
			return false;
		int m = matrix.length;
		int n = matrix[0].length;

		int beg = 0;
		int end = m * n - 1;

		while (beg <= end) {
			int mid = beg + (end - beg) / 2;
			int midX = mid / n;
			int midY = mid % n;

			if (matrix[midX][midY] == target)
				return true;
			if (matrix[midX][midY] > target)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		return false;
	}

	public boolean searchMatrix3(int[][] matrix, int target) {
		if (matrix.length == 0)
			return false;
		int beg = 0;
		int end = matrix.length - 1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (matrix[mid][0] == target)
				return true;
			if (matrix[mid][0] > target)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		int row = end;
		if (row < 0)
			return false;

		beg = 0;
		end = matrix[0].length - 1;

		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (matrix[row][mid] == target)
				return true;
			if (matrix[row][mid] > target)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		return false;
	}

	public int sumNumbers(TreeNode root) {
		return sumNumbers(root, 0);
	}

	public int sumNumbers(TreeNode root, int sum) {
		if (root == null)
			return 0;
		sum = sum * 10 + root.val;
		if (root.left == null && root.right == null)
			return sum;
		return sumNumbers(root.left, sum) + sumNumbers(root.right, sum);
	}

	public boolean hasPathSum(TreeNode root, int sum) {
		return hasPathSumUtil(root, sum, 0);
	}

	public boolean hasPathSumUtil(TreeNode root, int sum, int cursum) {
		if (root == null)
			return false;
		cursum += root.val;
		if (root.left == null && root.right == null && cursum == sum)
			return true;
		return hasPathSumUtil(root.left, sum, cursum)
				|| hasPathSumUtil(root.right, sum, cursum);
	}

	public boolean hasPathSum2(TreeNode root, int sum) {
		if (root == null)
			return false;
		sum -= root.val;
		if (root.left == null && root.right == null)
			return sum == 0;
		return hasPathSum2(root.left, sum) || hasPathSum2(root.right, sum);

	}

	public int minDepth(TreeNode root) {
		if (root == null)
			return 0;
		if (root.left == null)
			return minDepth(root.right) + 1;
		if (root.right == null)
			return minDepth(root.left) + 1;
		return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
	}

	public boolean isSymmetric(TreeNode root) {
		if (root == null)
			return true;
		return isSymmetric(root.left, root.right);
	}

	public boolean isSymmetric(TreeNode left, TreeNode right) {
		if (left == null && right == null)
			return true;
		if (left == null || right == null)
			return false;
		if (left.val != right.val)
			return false;
		return isSymmetric(left.left, right.right)
				&& isSymmetric(left.right, right.left);
	}

	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
		if (root == null || p == null || q == null)
			return null;
		if (p == root || q == root)
			return root;
		if (p.val > root.val && q.val > root.val)
			return lowestCommonAncestor(root.right, p, q);
		if (p.val < root.val && q.val < root.val)
			return lowestCommonAncestor(root.left, p, q);
		return root;
	}

	public int findPeakElement(int[] nums) {
		int beg = 0;
		int end = nums.length - 1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if ((mid == 0 || nums[mid] >= nums[mid - 1])
					&& (mid == nums.length - 1 || nums[mid] >= nums[mid + 1]))
				return mid;
			else if (mid > 0 && nums[mid] < nums[mid - 1])
				end = mid - 1;
			else
				beg = mid + 1;
		}
		return beg;
	}

	public List<String> binaryTreePaths(TreeNode root) {
		// Write your code here
		List<String> res = new ArrayList<String>();
		binaryTreePathsUtil(root, "", res);
		return res;
	}

	public void binaryTreePathsUtil(TreeNode root, String sol, List<String> res) {
		if (root == null)
			return;
		sol += root.val;
		if (root.left == null && root.right == null) {
			res.add(sol);
			return;
		}
		binaryTreePathsUtil(root.left, sol + "->", res);
		binaryTreePathsUtil(root.right, sol + "->", res);

	}
	
	public List<String> binaryTreePathsIt(TreeNode root) {
		List<String> res = new ArrayList<String>();
		if(root==null)
			return res;
		Queue<TreeNode> que=new LinkedList<TreeNode>();
		Queue<String> path = new LinkedList<String>();
		que.offer(root);
		path.offer(root.val+"");
		
		while(!que.isEmpty()){
			TreeNode cur=que.poll();
			String p=path.poll();
			if(cur.left==null&&cur.right==null){
				res.add(p);
			}
			if(cur.left!=null){
				que.offer(cur.left);
				path.offer(p+"->"+cur.left.val);
			}
			if(cur.right!=null){
				que.offer(cur.right);
				path.offer(p+"->"+cur.right.val);
			}
		}
		return res;
	}

	public int maxCoins(int[] nums) {
		int n = nums.length;
		if (n == 0)
			return 0;
		int[] balloons = new int[n + 2];

		for (int i = 1; i <= n; i++) {
			balloons[i] = nums[i - 1];
		}
		balloons[0] = balloons[n + 1] = 1;
		int[][] dp = new int[n + 2][n + 2];

		for (int k = 2; k <= n + 1; k++) {
			for (int left = 0; left < n - k + 2; left++) {
				int right = left + k;
				for (int i = left + 1; i < right; i++) {
					dp[left][right] = Math.max(dp[left][right], balloons[left]
							* balloons[i] * balloons[right] + dp[left][i]
							+ dp[i][right]);
				}
			}
		}
		return dp[0][n + 1];
	}

	public int maxProduct(int[] nums) {
		if (nums.length == 0)
			return 0;
		int localMin = nums[0];
		int localMax = nums[0];
		int global = nums[0];

		for (int i = 1; i < nums.length; i++) {
			int t = localMax;
			localMax = Math.max(Math.max(localMax * nums[i], nums[i]), localMin
					* nums[i]);
			localMin = Math.min(Math.min(localMin * nums[i], nums[i]), t
					* nums[i]);
			global = Math.max(global, localMax);
		}
		return global;
	}

	public int maxSubArray(int[] nums) {
		if (nums.length == 0)
			return 0;
		int localMax = nums[0];
		int global = nums[0];

		for (int i = 1; i < nums.length; i++) {
			localMax = Math.max(localMax + nums[i], nums[i]);
			global = Math.max(global, localMax);
		}
		return global;
	}

	public int maxProfit(int[] prices) {
		if (prices.length < 2)
			return 0;
		int lowest = prices[0];
		int maxProfit = 0;

		for (int i = 1; i < prices.length; i++) {
			lowest = Math.min(lowest, prices[i]);
			maxProfit = Math.max(maxProfit, prices[i] - lowest);
		}
		return maxProfit;
	}

	public int maxProfit2(int[] prices) {
		if (prices.length < 2)
			return 0;
		int profit = 0;
		for (int i = 1; i < prices.length; i++) {
			if (prices[i] - prices[i - 1] > 0)
				profit += prices[i] - prices[i - 1];
		}
		return profit;
	}

	public int maxProfit3(int[] prices) {
		if (prices.length < 2)
			return 0;
		int n = prices.length;
		int[] left = new int[n];
		int[] right = new int[n];
		int lowest = prices[0];
		for (int i = 1; i < prices.length; i++) {
			left[i] = Math.max(left[i - 1], prices[i] - lowest);
			lowest = Math.min(lowest, prices[i]);
		}
		int highest = prices[n - 1];
		for (int i = n - 2; i >= 0; i--) {
			right[i] = Math.max(right[i + 1], highest - prices[i]);
			highest = Math.max(highest, prices[i]);
		}
		int max = 0;

		for (int i = 0; i < n; i++) {
			if (left[i] + right[i] > max) {
				max = left[i] + right[i];
			}
		}
		return max;
	}

	public int search(int[] nums, int target) {
		int beg = 0;
		int end = nums.length - 1;

		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (nums[mid] == target)
				return mid;
			if (nums[beg] <= nums[mid]) {
				if (nums[beg] <= target && target < nums[mid]) {
					end = mid - 1;
				} else {
					beg = mid + 1;
				}
			} else {
				if (nums[mid] < target && target <= nums[end]) {
					beg = mid + 1;
				} else {
					end = mid - 1;
				}
			}
		}
		return -1;
	}

	public int divide(int dividend, int divisor) {
		boolean neg = false;
		boolean overflow = false;
		if (dividend < 0 && divisor > 0 || dividend > 0 && divisor < 0)
			neg = true;
		int res = 0;

		long a = Math.abs((long) dividend);
		long b = Math.abs((long) divisor);

		while (a >= b) {
			int shift = 0;
			while (a >= (b << shift)) {
				shift++;
			}
			shift--;
			a -= b << shift;
			if (res > Integer.MAX_VALUE - (1 << shift)) {
				overflow = true;
				break;
			}
			res += 1 << shift;
		}
		if (overflow) {
			if (neg)
				return Integer.MIN_VALUE;
			return Integer.MAX_VALUE;
		}
		return neg ? -res : res;
	}

	public int strStr(String haystack, String needle) {
		if (haystack.length() < needle.length())
			return -1;
		for (int i = 0; i < haystack.length() - needle.length() + 1; i++) {
			int j = 0;
			for (; j < needle.length(); j++) {
				if (needle.charAt(j) != haystack.charAt(i + j))
					break;
			}
			if (j == needle.length())
				return i;
		}
		return -1;
	}

	// dp[i]表示从s[i]到s[s.length – 1]最长的有效匹配括号子串长度
	public int longestValidParentheses(String s) {
		int n = s.length();
		int[] dp = new int[n];// dp[i] denotes the longest parenthesis length
								// from i to s.length()
		int maxLen = 0;

		for (int i = n - 2; i >= 0; i--) {
			if (s.charAt(i) == '(') {
				int j = i + dp[i + 1] + 1;
				if (j < s.length() && s.charAt(j) == ')') {
					dp[i] = dp[i + 1] + 2;
					if (j + 1 < s.length()) {
						dp[i] += dp[j + 1];
					}
					maxLen = Math.max(maxLen, dp[i]);
				}
			}
		}
		return maxLen;
	}

	public int longestValidParentheses2(String s) {
		int n = s.length();
		if (n < 2)
			return 0;
		Stack<Integer> stk = new Stack<Integer>();
		int max = 0;
		int last = -1;
		for (int i = 0; i < n; i++) {
			if (s.charAt(i) == '(')
				stk.push(i);
			else {
				if (stk.isEmpty())
					last = i;
				else {
					stk.pop();
					if (stk.isEmpty())
						max = Math.max(max, i - last);
					else
						max = Math.max(max, i - stk.peek());
				}
			}
		}
		return max;
	}

	public int searchInsert(int[] nums, int target) {
		int beg = 0;
		int end = nums.length - 1;

		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (nums[mid] == target)
				return mid;
			if (nums[mid] > target)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		return beg;
	}

	// testing
	public boolean isBadVersion(int n) {
		return true;
	}

	public int firstBadVersion(int n) {
		int beg = 1;
		int end = n;
		while (beg <= end) {
			int mid = beg + (end - beg) / 2;
			if (isBadVersion(mid))
				end = mid - 1;
			else
				beg = mid + 1;
		}
		return beg;
	}
	
	

	public int[] searchRange(int[] nums, int target) {
		int[] res = { -1, -1 };
		int beg = 0;
		int end = nums.length - 1;

		int index = -1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (nums[mid] == target) {
				index = mid;
				break;
			} else if (nums[mid] > target)
				end = mid - 1;
			else
				beg = mid + 1;
		}

		if (index == -1)
			return res;
		beg = 0;
		end = index;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (nums[mid] < target)
				beg = mid + 1;
			else
				end = mid - 1;
		}
		res[0] = beg;

		beg = index;
		end = nums.length - 1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (nums[mid] > target)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		res[1] = end;
		return res;
	}
	
	public int[] searchRange2(int[] nums, int target) {
		int[] res={-1,-1};
		int beg=0, end=nums.length-1;
		while(beg<end){
			int mid=(beg+end)/2;
			if(nums[mid]<target)
				beg=mid+1;
			else
				end=mid;
		}
		if(nums[beg]!=target)
			return res;
		res[0]=beg;
		end=nums.length-1;
		while(beg<end){
			int mid=(beg+end)/2+1;
			if(nums[mid]>target)
				end=mid-1;
			else
				beg=mid;
		}
		res[1]=end;
		return res;
	}

	public List<String> summaryRanges(int[] nums) {
		List<String> res = new ArrayList<String>();
		if (nums.length == 0)
			return res;
		int start = nums[0];

		for (int i = 1; i < nums.length; i++) {
			if (nums[i] != nums[i - 1] + 1) {
				if (nums[i - 1] == start) {
					res.add(start + "");
				} else {
					res.add(start + "->" + nums[i - 1]);
				}
				start = nums[i];
			}
		}
		if (nums[nums.length - 1] == start) {
			res.add(start + "");
		} else {
			res.add(start + "->" + nums[nums.length - 1]);
		}
		return res;
	}

	public List<String> summaryRanges2(int[] nums) {
		List<String> res = new ArrayList<String>();
		int start = 0, end = 0;

		while (end < nums.length) {
			if (end + 1 < nums.length && nums[end + 1] == nums[end] + 1) {
				end++;
			} else {
				if (start == end) {
					res.add(nums[start] + "");
				} else {
					res.add(nums[start] + "->" + nums[end]);
				}
				start = ++end;
			}
		}
		return res;
	}

	public int majorityElement(int[] nums) {
		int major = nums[0];
		int count = 1;
		for (int num : nums) {
			if (num == major)
				count++;
			else
				count--;
			if (count == 0) {
				major = num;
				count++;
			}
		}
		return major;
	}

	public List<Integer> majorityElement2(int[] nums) {
		List<Integer> res = new ArrayList<Integer>();
		int major1 = 0, major2 = 0, count1 = 0, count2 = 0;
		for (int num : nums) {
			// if (count1 == 0 || num == major1) {
			// major1 = num;
			// count1++;
			// } else if (count2 == 0 || num == major2) {
			// major2 = num;
			// count2++;
			// } else {
			// count1--;
			// count2--;
			// }
			if (num == major1)
				count1++;
			else if (num == major2)
				count2++;
			else if (count1 == 0) {
				major1 = num;
				count1 = 1;
			} else if (count2 == 0) {
				major2 = num;
				count2 = 1;
			} else {
				count1--;
				count2--;
			}
		}

		count1 = count2 = 0;
		for (int num : nums) {
			if (num == major1)
				count1++;
			if (num == major2)
				count2++;
		}
		if (count1 > nums.length / 3)
			res.add(major1);
		if (major2 != major1 && count2 > nums.length / 3)
			res.add(major2);

		return res;
	}

	// MAJORITY ELEMENTS 2
	public List<Integer> majorityElement_back(int[] nums) {
		List<Integer> res = new ArrayList<Integer>();
		int major1 = 0;
		int major2 = 0;
		int count1 = 0;
		int count2 = 0;
		for (int num : nums) {
			if (count1 == 0) {
				major1 = num;
			} else if (count2 == 0) {
				major2 = num;
			}
			if (num == major1)
				count1++;
			else if (num == major2)
				count2++;
			else {
				count1--;
				count2--;
			}
		}

		count1 = count2 = 0;
		for (int num : nums) {
			if (num == major1)
				count1++;
			else if (num == major2)
				count2++;
		}
		if (count1 > nums.length / 3)
			res.add(major1);
		if (count2 > nums.length / 3)
			res.add(major2);

		return res;
	}

	public void moveZeroes(int[] nums) {
		int beg = 0;
		int end = 0;
		while (end < nums.length) {
			if (nums[end] != 0) {
				nums[beg++] = nums[end++];
			} else {
				end++;
			}
		}
		for (int i = beg; i < nums.length; i++) {
			nums[i] = 0;
		}
	}

	public void moveZeroes2(int[] nums) {
		int j = 0;
		for (int i = 0; i < nums.length; i++) {
			if (nums[i] != 0) {
				int t = nums[j];
				nums[j] = nums[i];
				nums[i] = t;
				j++;
			}
		}
	}

	public ListNode rotateRight(ListNode head, int k) {
		if (head == null || head.next == null || k == 0)
			return head;
		int len = 0;
		ListNode node = head;
		while (node != null) {
			len++;
			node = node.next;
		}
		k = k % len;
		if (k == 0)
			return head;
		ListNode fast = head;
		for (int i = 0; i < k; i++) {
			fast = fast.next;
		}

		ListNode slow = head;
		while (fast.next != null) {
			fast = fast.next;
			slow = slow.next;
		}
		fast.next = head;
		ListNode next = slow.next;
		slow.next = null;
		return next;
	}

	public ListNode rotateRight2(ListNode head, int k) {
		if (head == null || head.next == null || k == 0)
			return head;
		ListNode node = head;
		int len = 1;
		while (node.next != null) {
			len++;
			node = node.next;
		}
		node.next = head;// form a loop
		k = k % len;
		for (int i = 0; i < len - k; i++) {
			node = node.next;
		}
		ListNode newHead = node.next; // find the break point, break it.
		node.next = null;
		return newHead;
	}

	public void rotate(int[] nums, int k) {
		int n = nums.length;
		k = k % n;
		reverseArray(nums, 0, n - k - 1);
		reverseArray(nums, n - k, n - 1);
		reverseArray(nums, 0, n - 1);
	}

	public void reverseArray(int[] nums, int beg, int end) {
		while (beg < end) {
			int t = nums[beg];
			nums[beg++] = nums[end];
			nums[end--] = t;
		}
	}

	public boolean containsDuplicate(int[] nums) {
		Set<Integer> map = new HashSet<Integer>();

		for (int num : nums) {
			if (map.contains(num))
				return true;
			map.add(num);
		}
		return false;
	}

	public boolean containsNearbyDuplicate(int[] nums, int k) {
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < nums.length; i++) {
			if (map.containsKey(nums[i])) {
				if (i - map.get(nums[i]) <= k)
					return true;
			}
			map.put(nums[i], i);
		}
		return false;
	}

	public boolean containsNearbyDuplicate_BF(int[] nums, int k) {

		for (int i = 0; i < nums.length; i++) {
			for (int j = 1; j <= k; j++) {
				if (nums[i] == nums[i + j])
					return true;
			}
		}
		return false;
	}

	public String reverseWords(String s) {
		String[] strs = s.trim().split(" ");
		StringBuilder sb = new StringBuilder();
		for (int i = strs.length - 1; i >= 0; i--) {
			if (!strs[i].isEmpty()) {
				sb.append(strs[i] + " ");
			}
		}
		return sb.toString().trim();
	}

	public int findMin(int[] nums) {
		return findMinUtil(nums, 0, nums.length - 1);
	}

	public int findMinUtil(int[] nums, int left, int right) {
		if (left == right)
			return nums[left];
		if (left == right - 1)
			return Math.min(nums[left], nums[right]);
		int mid = left + (right - left) / 2;
		if (nums[left] < nums[right])
			return nums[left];
		else if (nums[mid] > nums[left])
			return findMinUtil(nums, mid, right);
		return findMinUtil(nums, left, mid);
	}

	// (1) A[mid] < A[end]：A[mid : end] sorted => min不在A[mid+1 : end]中
	// 搜索A[start : mid]
	// (2) A[mid] > A[end]：A[start : mid] sorted且又因为该情况下A[end]<A[start] =>
	// min不在A[start : mid]中
	// 搜索A[mid+1 : end]
	// (3) base case：
	// a. start = end，必然A[start]为min，为搜寻结束条件。
	// b. start + 1 = end，此时A[mid] = A[start]，而min = min(A[mid],
	// A[end])。而这个条件可以合并到(1)和(2)中。

	public int findMin2(int[] nums) {
		int beg = 0;
		int end = nums.length - 1;
		while (beg < end) {
			int mid = beg + (end - beg) / 2;
			if (nums[mid] < nums[end])
				end = mid;
			else
				beg = mid + 1;
		}
		return nums[beg];
	}

	// 和Search in Rotated Sorted Array II这题换汤不换药。同样当A[mid] =
	// A[end]时，无法判断min究竟在左边还是右边。
	//
	// 3 1 2 3 3 3 3
	// 3 3 3 3 1 2 3
	//
	// 但可以肯定的是可以排除A[end]：因为即使min = A[end]，由于A[end] =
	// A[mid]，排除A[end]并没有让min丢失。所以增加的条件是：
	//
	// A[mid] = A[end]：搜索A[start : end-1]

	public int findMinDup(int[] nums) {
		int beg = 0;
		int end = nums.length - 1;

		while (beg < end) {
			int mid = beg + (end - beg) / 2;
			if (nums[mid] < nums[end])
				end = mid;
			else if (nums[mid] > nums[end])
				beg = mid + 1;
			else
				end--;
		}
		return nums[beg];
	}

	public int evalRPN(String[] tokens) {
		Stack<Integer> stk = new Stack<Integer>();

		for (int i = 0; i < tokens.length; i++) {
			String op = tokens[i];
			if (!op.equals("+") && !op.equals("-") && !op.equals("*")
					&& !op.equals("/")) {
				int operand = Integer.parseInt(op);
				stk.push(operand);
			} else {
				int operand1 = stk.pop();
				int operand2 = stk.pop();
				if (op.equals("+"))
					stk.push(operand2 + operand1);
				else if (op.equals("-"))
					stk.push(operand2 - operand1);
				else if (op.equals("*"))
					stk.push(operand2 * operand1);
				else
					stk.push(operand2 / operand1);
			}
		}
		return stk.pop();
	}

	public int evalRPN2(String[] tokens) {
		Stack<Integer> stk = new Stack<Integer>();

		for (int i = 0; i < tokens.length; i++) {
			String op = tokens[i];
			if (!op.equals("+") && !op.equals("-") && !op.equals("*")
					&& !op.equals("/")) {
				int operand = Integer.parseInt(op);
				stk.push(operand);
			} else {
				int operand1 = stk.pop();
				int operand2 = stk.pop();
				switch (op) {
				case "+":
					stk.push(operand2 + operand1);
					break;
				case "-":
					stk.push(operand2 - operand1);
					break;
				case "*":
					stk.push(operand2 * operand1);
					break;
				case "/":
					stk.push(operand2 / operand1);
					break;
				}
			}
		}
		return stk.pop();
	}

	public List<List<Integer>> pathSum(TreeNode root, int sum) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> path = new ArrayList<Integer>();
		pathSumUtil(root, sum, path, res, 0);
		return res;
	}

	public void pathSumUtil(TreeNode root, int sum, List<Integer> path,
			List<List<Integer>> res, int cursum) {
		if (root == null)
			return;
		cursum += root.val;
		path.add(root.val);
		if (cursum == sum && root.left == null && root.right == null) {
			List<Integer> out = new ArrayList<Integer>(path);
			res.add(out);
		}
		pathSumUtil(root.left, sum, path, res, cursum);
		pathSumUtil(root.right, sum, path, res, cursum);
		cursum -= root.val;
		path.remove(path.size() - 1);
	}

	public String largestNumber(int[] nums) {
		String[] strs = new String[nums.length];
		for (int i = 0; i < nums.length; i++) {
			strs[i] = "" + nums[i];
		}
		Arrays.sort(strs, new Comparator<String>() {
			@Override
			public int compare(String s1, String s2) {
				return (s2 + s1).compareTo(s1 + s2);
			}
		});
		if (strs[0].equals("0"))
			return "0";
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < strs.length; i++) {
			sb.append(strs[i]);
		}
		return sb.toString();
	}

	public int compareVersion(String version1, String version2) {
		String[] v1 = version1.split("\\.");
		String[] v2 = version2.split("\\.");
		int i = 0;
		for (; i < v1.length && i < v2.length; i++) {
			if (Integer.parseInt(v1[i]) > Integer.parseInt(v2[i]))
				return 1;
			else if (Integer.parseInt(v1[i]) < Integer.parseInt(v2[i]))
				return -1;
		}
		while (i < v1.length) {
			if (Integer.parseInt(v1[i++]) != 0)
				return 1;
		}
		while (i < v2.length) {
			if (Integer.parseInt(v2[i++]) != 0)
				return -1;
		}
		return 0;
	}

	public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
		if (headA == null || headB == null)
			return null;
		ListNode node = headA;
		int lenA = 0;
		while (node != null) {
			lenA++;
			node = node.next;
		}
		node = headB;
		int lenB = 0;
		while (node != null) {
			lenB++;
			node = node.next;
		}
		node = lenA > lenB ? headA : headB;
		for (int i = 0; i < Math.abs(lenA - lenB); i++) {
			node = node.next;
		}
		ListNode node2 = lenA > lenB ? headB : headA;
		while (node != node2) {
			node = node.next;
			node2 = node2.next;
		}
		return node;
	}

	public ListNode getIntersectionNode2(ListNode headA, ListNode headB) {
		if (headA == null || headB == null)
			return null;

		ListNode a = headA;
		ListNode b = headB;
		while (a != b) {
			// for the end of first iteration, we just reset the pointer to the
			// head of another linkedlist
			a = a == null ? headB : a.next;
			b = b == null ? headA : b.next;
		}
		return a;
	}

	public ListNode deleteDuplicates(ListNode head) {
		if (head == null || head.next == null)
			return head;

		ListNode node = head.next;
		ListNode pre = head;
		while (node != null) {
			if (node.val == pre.val) {
				node = node.next;
			} else {
				pre.next = node;
				pre = node;
				node = node.next;
			}
		}
		pre.next = null;
		return head;
	}

	public void deleteNode(ListNode node) {
		if (node == null)
			return;
		ListNode pnext = node.next;
		node.val = pnext.val;
		node.next = pnext.next;
	}

	public ListNode removeElements(ListNode head, int val) {
		if (head == null)
			return null;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode pre = dummy;
		ListNode cur = head;

		while (cur != null) {
			if (cur.val == val) {
				cur = cur.next;
				pre.next = cur;
			} else {
				pre = cur;
				cur = cur.next;
			}
		}
		return dummy.next;
	}

	public ListNode partition(ListNode head, int x) {
		if (head == null) {
			return null;
		}

		ListNode leftDummy = new ListNode(0);
		ListNode rightDummy = new ListNode(0);
		ListNode left = leftDummy, right = rightDummy;

		while (head != null) {
			if (head.val < x) {
				left.next = head;
				left = head;
			} else {
				right.next = head;
				right = head;
			}
			head = head.next;
		}

		right.next = null;
		left.next = rightDummy.next;
		return leftDummy.next;
	}

	public ListNode reverseBetween(ListNode head, int m, int n) {
		if (head == null)
			return null;
		ListNode dummy = new ListNode(0);
		dummy.next = head;

		ListNode pPre = dummy;
		ListNode cur = head;
		for (int i = 0; i < m - 1; i++) {
			pPre = cur;
			cur = cur.next;
		}
		ListNode pre = cur;
		ListNode start = cur;
		cur = cur.next;
		for (int i = m; i < n; i++) {
			ListNode pnext = cur.next;
			cur.next = pre;
			pre = cur;
			cur = pnext;
		}
		pPre.next = pre;
		start.next = cur;
		return dummy.next;
	}

	public ListNode reverseList(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode cur = head;
		ListNode pre = null;
		while (cur != null) {
			ListNode pnext = cur.next;
			cur.next = pre;
			pre = cur;
			cur = pnext;
		}
		return pre;
	}

	public ListNode reverseList_Recursive(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode pnext = head.next;
		ListNode t = reverseList(pnext);
		pnext.next = head;
		head.next = null;
		return t;
	}

	public ListNode reverseList_General(ListNode head) {
		// solution 3
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;

		ListNode last = head;
		ListNode cur = head.next;
		ListNode pre = dummy;
		while (cur != null) {
			last.next = cur.next;
			cur.next = pre.next;
			pre.next = cur;
			cur = last.next;
		}
		return dummy.next;
	}

	public String convertToTitle(int n) {
		StringBuilder sb = new StringBuilder();
		while (n > 0) {
			n--;
			char c = (char) ('A' + n % 26);
			sb.insert(0, c);
			n /= 26;
		}
		return sb.toString();
	}

	public int titleToNumber(String s) {
		int res = 0;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			res = res * 26 + (c - 'A' + 1);
		}
		return res;
	}

	public boolean isPowerOfTwo(int n) {
		return n > 0 && (n & (n - 1)) == 0;
	}

	public int hammingWeight(int n) {
		int count = 0;
		for (int i = 0; i < 32; i++) {
			if ((n & (1 << i)) != 0) {
				count++;
			}
		}
		return count;
	}

	public int reverseBits(int n) {
		int res = 0;
		for (int i = 0; i < 32; i++) {
			int b = (n >> i) & 1;
			res = res * 2 + b;
		}
		return res;
	}

	public List<List<Integer>> subsets(int[] nums) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		Arrays.sort(nums);
		subsetsUtil(nums, 0, sol, res);
		return res;
	}

	public void subsetsUtil(int[] nums, int cur, List<Integer> sol,
			List<List<Integer>> res) {
		List<Integer> out = new ArrayList<Integer>(sol);
		res.add(out);
		if (cur == nums.length)
			return;
		for (int i = cur; i < nums.length; i++) {
			sol.add(nums[i]);
			subsetsUtil(nums, i + 1, sol, res);
			sol.remove(sol.size() - 1);
		}
	}

	public List<List<Integer>> subsetsWithDup(int[] nums) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (nums.length == 0)
			return res;
		Arrays.sort(nums);
		boolean[] used = new boolean[nums.length];
		List<Integer> sol = new ArrayList<Integer>();
		subsetsWithDup(0, nums, used, sol, res, 0);
		return res;
	}

	public void subsetsWithDup(int dep, int[] nums, boolean[] used,
			List<Integer> sol, List<List<Integer>> res, int cur) {
		res.add(new ArrayList<Integer>(sol));
		if (dep == nums.length)
			return;
		for (int i = cur; i < nums.length; i++) {
			if (!used[i]) {
				if (i != 0 && nums[i] == nums[i - 1] && !used[i - 1])
					continue;
				sol.add(nums[i]);
				used[i] = true;
				subsetsWithDup(dep + 1, nums, used, sol, res, i + 1);
				sol.remove(sol.size() - 1);
				used[i] = false;
			}
		}
	}

	public List<List<Integer>> subsetsWithDup2(int[] nums) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (nums.length == 0)
			return res;
		Arrays.sort(nums);
		List<Integer> sol = new ArrayList<Integer>();
		subsetsWithDup(nums, sol, res, 0);
		return res;
	}

	public void subsetsWithDup(int[] nums, List<Integer> sol,
			List<List<Integer>> res, int cur) {
		res.add(new ArrayList<Integer>(sol));

		for (int i = cur; i < nums.length; i++) {
			if (i != cur && nums[i] == nums[i - 1])
				continue;
			sol.add(nums[i]);
			subsetsWithDup(nums, sol, res, i + 1);
			sol.remove(sol.size() - 1);
		}
	}

	public List<List<Integer>> subsetsWithDup3(int[] nums) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (nums.length == 0)
			return res;
		Arrays.sort(nums);
		List<Integer> sol = new ArrayList<Integer>();
		for (int len = 1; len <= nums.length; len++) {
			dfsSubsets(len, nums, sol, res, 0);
		}
		res.add(new ArrayList<Integer>());
		return res;
	}

	public void dfsSubsets(int len, int[] nums, List<Integer> sol,
			List<List<Integer>> res, int cur) {
		if (sol.size() == len) {
			res.add(new ArrayList<Integer>(sol));
			return;
		}

		for (int i = cur; i < nums.length; i++) {
			sol.add(nums[i]);
			dfsSubsets(len, nums, sol, res, i + 1);
			sol.remove(sol.size() - 1);
			while (i < nums.length - 1 && nums[i] == nums[i + 1])
				// skip the duplicates
				i++;
		}
	}

	public void nextPermutation(int[] nums) {
		int index = -1;
		for (int i = 0; i < nums.length - 1; i++) {
			if (nums[i] < nums[i + 1]) {
				index = i;
			}
		}
		if (index == -1) {
			Arrays.sort(nums);
			return;
		}
		int pos = index + 1;
		for (int i = index; i < nums.length; i++) {
			if (nums[i] > nums[index]) {
				pos = i;
			}
		}

		swap(nums, index, pos);

		int beg = index + 1;
		int end = nums.length - 1;
		while (beg < end) {
			swap(nums, beg++, end--);
		}
	}

	public void swap(int[] nums, int i, int j) {
		int t = nums[i];
		nums[i] = nums[j];
		nums[j] = t;
	}

	public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (root == null)
			return res;
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		List<Integer> level = new ArrayList<Integer>();
		int curLevel = 0;
		int nextLevel = 0;
		que.add(root);
		curLevel++;
		boolean left2Right = true;
		while (!que.isEmpty()) {
			TreeNode top = que.remove();
			level.add(top.val);
			curLevel--;
			if (top.left != null) {
				que.add(top.left);
				nextLevel++;
			}
			if (top.right != null) {
				que.add(top.right);
				nextLevel++;
			}
			if (curLevel == 0) {
				if (!left2Right) {
					Collections.reverse(level);
				}
				res.add(level);
				level = new ArrayList<Integer>();
				left2Right = !left2Right;
				curLevel = nextLevel;
				nextLevel = 0;
			}
		}
		return res;
	}

	public List<List<Integer>> levelOrder(TreeNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (root == null)
			return res;
		List<Integer> level = new ArrayList<Integer>();

		Queue<TreeNode> que = new LinkedList<TreeNode>();
		int curlevel = 0, nextlevel = 0;
		que.add(root);
		curlevel++;
		while (!que.isEmpty()) {
			TreeNode top = que.remove();
			level.add(top.val);
			curlevel--;
			if (top.left != null) {
				que.add(top.left);
				nextlevel++;
			}
			if (top.right != null) {
				que.add(top.right);
				nextlevel++;
			}
			if (curlevel == 0) {
				curlevel = nextlevel;
				nextlevel = 0;
				res.add(level);
				level = new ArrayList<Integer>();
			}
		}
		return res;
	}

	public int findKthLargest(int[] nums, int k) {
		return quickSelect(nums, 0, nums.length - 1, k);
	}

	public int quickSelect(int[] nums, int left, int right, int k) {
		int pivot = left;
		int beg = left + 1, end = right;

		while (beg <= end) {
			while (beg <= end && nums[beg] >= nums[pivot]) {
				beg++;
			}
			while (beg <= end && nums[end] < nums[pivot]) {
				end--;
			}
			if (beg < end) {
				swap(nums, beg++, end--);
			}
		}
		swap(nums, pivot, end);
		if (end == k - 1)
			return nums[end];
		else if (end > k - 1)
			return quickSelect(nums, left, end - 1, k);
		return quickSelect(nums, end + 1, right, k);
	}

	public int findKthLargest2(int[] nums, int k) {
		int n = nums.length;
		return findKthSmallestUtil(nums, 0, n - 1, n - k + 1);
	}

	// return the index of the kth smallest number
	public int findKthSmallestUtil(int[] nums, int lo, int hi, int k) {
		// use quick sort's idea
		// put nums that are <= pivot to the left
		// put nums that are > pivot to the right
		int pivot = nums[hi];
		int i = lo, j = hi;

		while (i < j) {
			if (nums[i++] > pivot) {
				swap(nums, --i, --j);
			}
		}
		swap(nums, i, hi);
		// count the nums that are <= pivot from lo
		int m = i - lo + 1;
		if (m == k)
			return nums[i];
		// pivot is too big, so it must be on the left
		else if (m > k)
			return findKthSmallestUtil(nums, lo, i - 1, k);
		// pivot is too small, so it must be on the right
		return findKthSmallestUtil(nums, i + 1, hi, k - m);
	}

	public int findKthLargest3(int[] nums, int k) {
		int n = nums.length;
		return quickSelect(nums, 0, n - 1, k);
	}

	public int findKthLargestUtil2(int[] nums, int lo, int hi, int k) {
		int i = lo, j = hi, pivot = nums[hi];
		while (i < j) {
			if (nums[i++] < pivot) {
				swap(nums, --i, --j);
			}
		}
		swap(nums, i, hi);
		int m = i - lo + 1;
		if (m == k)
			return nums[i];
		else if (m > k - 1)
			return findKthLargestUtil2(nums, lo, i - 1, k);
		return findKthLargestUtil2(nums, i + 1, hi, k - m);
	}

	public List<Integer> diffWaysToCompute(String input) {
		List<Integer> res = new ArrayList<Integer>();
		if (input.length() == 0)
			return res;
		for (int i = 0; i < input.length(); i++) {
			char c = input.charAt(i);
			if (c == '+' || c == '-' || c == '*') {
				List<Integer> res1 = diffWaysToCompute(input.substring(0, i));
				List<Integer> res2 = diffWaysToCompute(input.substring(i + 1));

				for (int j = 0; j < res1.size(); j++) {
					for (int k = 0; k < res2.size(); k++) {
						if (c == '+')
							res.add(res1.get(j) + res2.get(k));
						else if (c == '-')
							res.add(res1.get(j) - res2.get(k));
						else
							res.add(res1.get(j) * res2.get(k));
					}
				}
			}
		}
		if (res.size() == 0)
			res.add(Integer.parseInt(input));
		return res;
	}

	public List<Integer> diffWaysToComputeDP(String input) {
		HashMap<String, List<Integer>> map = new HashMap<String, List<Integer>>();
		return diffWaysToComputeUtil(input, map);
	}

	public List<Integer> diffWaysToComputeUtil(String input,
			HashMap<String, List<Integer>> map) {
		List<Integer> res = new ArrayList<Integer>();
		if (input.length() == 0)
			return res;
		for (int i = 0; i < input.length(); i++) {
			List<Integer> res1, res2;
			char c = input.charAt(i);
			if (c == '+' || c == '-' || c == '*') {
				String s1 = input.substring(0, i);
				String s2 = input.substring(i + 1);
				if (map.containsKey(s1)) {
					res1 = map.get(s1);
				} else {
					res1 = diffWaysToComputeUtil(s1, map);
				}

				if (map.containsKey(s2)) {
					res2 = map.get(s2);
				} else {
					res2 = diffWaysToComputeUtil(s2, map);
				}
				for (int j = 0; j < res1.size(); j++) {
					for (int k = 0; k < res2.size(); k++) {
						if (c == '+')
							res.add(res1.get(j) + res2.get(k));
						else if (c == '-')
							res.add(res1.get(j) - res2.get(k));
						else
							res.add(res1.get(j) * res2.get(k));
					}
				}
			}
		}
		if (res.size() == 0)
			res.add(Integer.parseInt(input));
		map.put(input, res);
		return res;
	}

	public List<String> addOperators(String num, int target) {
		List<String> res = new ArrayList<String>();
		if (num.length() == 0)
			return res;
		addOperatorsUtil(num, "", res, target, 0, 0, 0);
		return res;
	}

	public void addOperatorsUtil(String num, String sol, List<String> res,
			int target, int curPos, long curSum, long lastVal) {
		if (curPos == num.length() && curSum == target) {
			res.add(sol);
			return;
		}

		for (int i = curPos; i < num.length(); i++) {
			String s = num.substring(curPos, i + 1);
			if (s.length() > 1 && s.charAt(0) == '0')
				return;
			long curNum = Long.parseLong(s);
			if (curPos == 0) {
				addOperatorsUtil(num, sol + "" + curNum, res, target, i + 1,
						curNum, curNum);
			} else {
				addOperatorsUtil(num, sol + "+" + curNum, res, target, i + 1,
						curSum + curNum, curNum);
				addOperatorsUtil(num, sol + "-" + curNum, res, target, i + 1,
						curSum - curNum, -curNum);
				addOperatorsUtil(num, sol + "*" + curNum, res, target, i + 1,
						curSum - lastVal + lastVal * curNum, lastVal * curNum);
			}
		}
	}

	public ListNode reverseKGroup(ListNode head, int k) {
		if (head == null || head.next == null || k == 1)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode pre = dummy, cur = head;
		int count = 0;
		while (cur != null) {
			count++;
			if (count % k == 0) {
				pre = reverseKNodes(pre, cur.next);
				cur = pre.next;
			} else {
				cur = cur.next;
			}
		}
		return dummy.next;
	}

	public ListNode reverseKNodes(ListNode pre, ListNode next) {
		ListNode last = pre.next;
		ListNode cur = last.next;

		while (cur != next) {
			last.next = cur.next;
			cur.next = pre.next;
			pre.next = cur;
			cur = last.next;
		}
		return last;
	}

	// Given 1->2->3->3->4->4->5, return 1->2->5.
	public ListNode deleteDuplicates2(ListNode head) {
		ListNode dummy = new ListNode(0);
		dummy.next = head;

		ListNode cur = head;
		ListNode pre = dummy;
		while (cur != null) {
			boolean dup = false;
			while (cur.next != null && cur.val == cur.next.val) {
				dup = true;
				cur = cur.next;
			}
			if (dup) {
				pre.next = cur.next;
			} else {
				pre.next = cur;
				pre = cur;
			}
			cur = cur.next;
		}
		return dummy.next;
	}

	// Given 1->2->3->3->4->4->5, return 1->2->5.
	public ListNode deleteDuplicates3(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode cur = head, pre = dummy;

		while (cur != null) {
			while (cur.next != null && cur.val == cur.next.val) {
				cur = cur.next;
			}
			if (pre.next == cur)
				pre = cur;
			else
				pre.next = cur.next;
			cur = cur.next;
		}
		return dummy.next;

	}

	public boolean hasCycle(ListNode head) {
		if (head == null || head.next == null)
			return false;
		ListNode fast = head, slow = head;

		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow)
				break;
		}
		return fast == slow ? true : false;
	}

	public ListNode detectCycle(ListNode head) {
		if (head == null || head.next == null)
			return null;
		ListNode fast = head, slow = head;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow)
				break;
		}
		if (fast == null || fast.next == null)
			return null;
		fast = head;
		while (fast != slow) {
			fast = fast.next;
			slow = slow.next;
		}
		return fast;
	}

	public ListNode detectCycle2(ListNode head) {
		ListNode slow = head;
		ListNode fast = head;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow) {
				// 找相遇点
				slow = head;
				while (slow != fast) {
					slow = slow.next;
					fast = fast.next;
				}
				return slow;
			}
		}
		return null;
	}

	public String fractionToDecimal(int numerator, int denominator) {
		if (numerator == 0 || denominator == 0)
			return "0";
		String res = "";
		if ((numerator < 0) ^ (denominator < 0))
			res += "-";
		long num = numerator;
		long denom = denominator;
		num = Math.abs(num);
		denom = Math.abs(denom);

		res += num / denom;
		long rem = num % denom;
		if (rem == 0)
			return res;
		res += ".";
		HashMap<Long, Integer> map = new HashMap<Long, Integer>();
		rem *= 10;
		while (rem > 0) {
			if (map.containsKey(rem)) {
				int index = map.get(rem);
				String s1 = res.substring(0, index);
				String s2 = res.substring(index);
				res = s1 + "(" + s2 + ")";
				return res;
			} else {
				map.put(rem, res.length());
				res += rem / denom;
				rem = rem % denom * 10;
			}
		}
		return res;
	}

	public int rob(int[] nums) {
		int m = nums.length;
		if (m == 0)
			return 0;
		int[] dp = new int[m];
		dp[0] = nums[0];
		if (m > 1)
			dp[1] = Math.max(nums[0], nums[1]);
		for (int i = 2; i < m; i++) {
			dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
		}
		return dp[m - 1];
	}

	public int rob2(int[] nums) {
		int m = nums.length;
		if (m == 0)
			return 0;
		if (m == 1)
			return nums[0];
		if (m == 2)
			return Math.max(nums[0], nums[1]);
		// 1. rob the first, not rob the last, 2 rob the last, not rob first
		return Math.max(robUtil(nums, 0, m - 2), robUtil(nums, 1, m - 1));
	}

	public boolean isSameTree(TreeNode p, TreeNode q) {
		if (p == null && q == null)
			return true;
		if (p == null || q == null)
			return false;
		if (p.val != q.val)
			return false;
		return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
	}

	public int robUtil(int[] nums, int left, int right) {
		int len = right - left + 1;
		int[] dp = new int[len];
		dp[0] = nums[left];
		dp[1] = Math.max(nums[left], nums[left + 1]);
		for (int i = 2; i < len; i++) {
			dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[left + i]);
		}
		return dp[len - 1];
	}

	public boolean isSubtree(TreeNode T, TreeNode S) {
		if (S == null)
			return true;
		if (T == null)
			return false;
		List<Integer> inT = convert2Inorder(T);
		List<Integer> inS = convert2Inorder(S);
		if (!isSublist(inT, inS))
			return false;
		List<Integer> preT = convert2Preorder(T);
		List<Integer> preS = convert2Preorder(S);
		return isSublist(preT, preS);
	}

	public List<Integer> convert2Inorder(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		return convert2Inorder(root, res);
	}

	public List<Integer> convert2Inorder(TreeNode root, List<Integer> res) {
		if (root == null)
			return new ArrayList<Integer>();
		List<Integer> left = convert2Inorder(root.left);
		res.addAll(left);
		res.add(root.val);
		List<Integer> right = convert2Inorder(root.right);
		res.addAll(right);
		return res;
	}

	public List<Integer> convert2Preorder(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		return convert2Preorder(root, res);
	}

	public List<Integer> convert2Preorder(TreeNode root, List<Integer> res) {
		if (root == null)
			return new ArrayList<Integer>();

		res.add(root.val);
		List<Integer> left = convert2Preorder(root.left);
		res.addAll(left);
		List<Integer> right = convert2Preorder(root.right);
		res.addAll(right);
		return res;
	}

	public boolean isSublist(List<Integer> l1, List<Integer> l2) {
		if (l1.size() < l2.size())
			return false;
		int i = 0, j = 0;
		while (i < l1.size() && j < l2.size()) {
			if (l1.get(i) != l2.get(j)) {
				i++;
				j = 0;
			} else {
				i++;
				j++;
			}
		}

		return j == l2.size();
	}

	public ListNode sortList(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode fast = head, slow = head;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			if (fast == null)
				break;
			slow = slow.next;
		}

		ListNode secondHalf = slow.next;
		slow.next = null;

		ListNode firstHalf = sortList(head);
		secondHalf = sortList(secondHalf);
		return mergeList(firstHalf, secondHalf);
	}

	public ListNode mergeList(ListNode head1, ListNode head2) {
		if (head1 == null || head2 == null)
			return head1 == null ? head2 : head1;
		ListNode dummy = new ListNode(0);
		ListNode pre = dummy;
		while (head1 != null && head2 != null) {
			if (head1.val < head2.val) {
				pre.next = head1;
				head1 = head1.next;
			} else {
				pre.next = head2;
				head2 = head2.next;
			}
			pre = pre.next;
		}
		if (head1 != null)
			pre.next = head1;
		if (head2 != null)
			pre.next = head2;
		return dummy.next;
	}

	public ListNode insertionSortList(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;

		ListNode last = head, cur = last.next;
		while (cur != null) {
			ListNode pre = dummy, iterator = dummy.next;
			while (iterator != cur && iterator.val < cur.val) {
				pre = iterator;
				iterator = iterator.next;
			}
			if (iterator != cur) {
				last.next = cur.next;
				pre.next = cur;
				cur.next = iterator;
			} else {
				last = cur;
			}
			cur = last.next;
		}
		return dummy.next;
	}

	public void sortColors(int[] nums) {
		if (nums.length < 2)
			return;
		int i = 0, j = nums.length - 1, k = nums.length - 1;

		while (i <= j) {
			if (nums[i] == 2) {
				nums[i] = nums[k];
				nums[k--] = 2;
				if (j > k)
					j--;
			} else if (nums[i] == 1) {
				nums[i] = nums[j];
				nums[j--] = 1;
			} else {
				i++;
			}
		}
	}

	public int kthSmallest(TreeNode root, int k) {
		int count = countNodes(root.left);
		if (count == k - 1)
			return root.val;
		else if (count > k - 1)
			return kthSmallest(root.left, k);
		else
			return kthSmallest(root.right, k - count - 1);
	}

	public int countNodes(TreeNode root) {
		if (root == null)
			return 0;
		return countNodes(root.left) + countNodes(root.right) + 1;
	}

	public List<Integer> inorderTraversal(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		if (root == null)
			return res;
		Stack<TreeNode> stk = new Stack<TreeNode>();
		TreeNode cur = root;
		while (cur != null) {
			stk.push(cur);
			cur = cur.left;
		}

		while (!stk.isEmpty()) {
			TreeNode top = stk.pop();
			res.add(top.val);
			if (top.right != null) {
				top = top.right;
				while (top != null) {
					stk.push(top);
					top = top.left;
				}
			}
		}
		return res;
	}

	public List<Integer> preorderTraversal(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		if (root == null)
			return res;
		Stack<TreeNode> stk = new Stack<TreeNode>();
		stk.push(root);

		while (!stk.isEmpty()) {
			TreeNode top = stk.pop();
			res.add(top.val);
			if (top.right != null) {
				stk.push(top.right);
			}
			if (top.left != null) {
				stk.push(top.left);
			}
		}
		return res;
	}

	public List<Integer> postorderTraversal(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		if (root == null)
			return res;
		Stack<TreeNode> stk = new Stack<TreeNode>();
		TreeNode cur = root;
		while (cur != null) {
			stk.push(cur);
			cur = cur.left;
		}
		TreeNode pre = null;
		while (!stk.isEmpty()) {
			TreeNode top = stk.peek();
			if (top.right != null && pre != top.right) {
				top = top.right;
				while (top != null) {
					stk.push(top);
					top = top.left;
				}
			} else {
				res.add(top.val);
				pre = stk.pop();
			}
		}
		return res;
	}

	public List<Integer> rightSideView(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		if (root == null)
			return res;
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		int curlevel = 0, nextlevel = 0;
		que.add(root);
		curlevel++;
		while (!que.isEmpty()) {
			TreeNode top = que.remove();
			curlevel--;
			if (top.left != null) {
				que.add(top.left);
				nextlevel++;
			}
			if (top.right != null) {
				que.add(top.right);
				nextlevel++;
			}
			if (curlevel == 0) {
				curlevel = nextlevel;
				nextlevel = 0;
				res.add(top.val);
			}
		}
		return res;
	}

	public List<Integer> rightSideViewDFS(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		rightSideView(root, res, 1);
		return res;
	}

	public void rightSideView(TreeNode root, List<Integer> res, int level) {
		if (root == null)
			return;
		if (res.size() < level) {
			res.add(root.val);
		}
		rightSideView(root.right, res, level + 1);
		rightSideView(root.left, res, level + 1);
	}

	public void connect(TreeLinkNode root) {
		if (root == null)
			return;
		if (root.left != null) {
			root.left.next = root.right;
		}
		if (root.right != null) {
			if (root.next != null)
				root.right.next = root.next.left;
		}

		connect(root.left);
		connect(root.right);
	}

	public void connectIterative(TreeLinkNode root) {
		TreeLinkNode cur = root;
		while (cur != null) {
			TreeLinkNode left = cur.left;
			while (cur != null) {// go right
				if (cur.left != null)
					cur.left.next = cur.right;
				if (cur.right != null && cur.next != null)
					cur.right.next = cur.next.left;
				cur = cur.next;
			}
			cur = left;// go next level
		}
	}

	public void connect2BFS(TreeLinkNode root) {
		if (root == null)
			return;
		Queue<TreeLinkNode> que = new LinkedList<TreeLinkNode>();
		int curlevel = 0, nextlevel = 0;
		que.add(root);
		curlevel++;
		while (!que.isEmpty()) {
			TreeLinkNode top = que.remove();
			curlevel--;
			if (top.left != null) {
				que.add(top.left);
				nextlevel++;
			}
			if (top.right != null) {
				que.add(top.right);
				nextlevel++;
			}
			if (curlevel == 0) {
				curlevel = nextlevel;
				nextlevel = 0;
				top.next = null;
			} else {
				top.next = que.peek();
			}
		}
	}

	public void connect2Iterative(TreeLinkNode root) {
		if (root == null)
			return;
		TreeLinkNode cur = root;
		while (cur != null) {
			TreeLinkNode next = null;
			TreeLinkNode pre = null;
			while (cur != null) { // go right
				if (next == null)
					next = cur.left == null ? cur.right : cur.left;
				if (cur.left != null) {
					if (pre != null)
						pre.next = cur.left;
					pre = cur.left;
				}

				if (cur.right != null) {
					if (pre != null)
						pre.next = cur.right;
					pre = cur.right;
				}
				cur = cur.next;
			}
			cur = next;// next level
		}
	}

	public void flatten(TreeNode root) {
		if (root == null)
			return;
		TreeNode right = root.right;
		flatten(root.left);
		root.right = root.left;
		root.left = null;
		TreeNode cur = root;
		while (cur.right != null) {
			cur = cur.right;
		}
		cur.right = right;
		flatten(right);
	}

	public void flattenIterative(TreeNode root) {
		if (root == null)
			return;
		TreeNode cur = root;
		while (cur != null) {
			if (cur.left != null) {
				TreeNode pre = cur.left;
				while (pre.right != null) {
					pre = pre.right;
				}
				pre.right = cur.right;
				cur.right = cur.left;
				cur.left = null;
			}
			cur = cur.right;
		}
	}

	public void flattenStack(TreeNode root) {
		if (root == null)
			return;
		Stack<TreeNode> stk = new Stack<TreeNode>();
		stk.push(root);
		TreeNode pre = null;
		while (!stk.isEmpty()) {
			TreeNode cur = stk.pop();
			if (cur.right != null)
				stk.push(cur.right);
			if (cur.left != null)
				stk.push(cur.left);
			if (pre != null) {
				pre.right = cur;
				pre.left = null;
			}
			pre = cur;
		}
	}

	// This problem can be converted as a overlap internal problem.
	// On the x-axis, there are (A,C) and (E,G);
	// on the y-axis, there are (F,H) and (B,D).
	// If they do not have overlap, the total area is the sum of 2 rectangle
	// areas.
	// If they have overlap, the total area should minus the overlap area.

	public int computeArea(int A, int B, int C, int D, int E, int F, int G,
			int H) {
		if (C <= E || G <= A || B >= H || F >= D)
			return (D - B) * (C - A) + (G - E) * (H - F);
		int left = Math.max(A, E);
		int right = Math.min(C, G);
		int top = Math.min(D, H);
		int bottom = Math.max(B, F);

		return (D - B) * (C - A) + (G - E) * (H - F) - (right - left)
				* (top - bottom);
	}

	public String addBinary(String a, String b) {
		if (a.isEmpty() || b.isEmpty())
			return a.isEmpty() ? b : a;
		int carry = 0;
		int i = a.length() - 1, j = b.length() - 1;

		StringBuilder sb = new StringBuilder();
		while (i >= 0 || j >= 0) {
			int v1 = 0, v2 = 0;
			if (i >= 0)
				v1 = a.charAt(i--) - '0';
			if (j >= 0)
				v2 = b.charAt(j--) - '0';
			int sum = v1 + v2 + carry;
			carry = sum / 2;
			sum = sum % 2;
			sb.insert(0, sum);
		}
		if (carry == 1)
			sb.insert(0, 1);
		return sb.toString();
	}

	public int[] plusOne(int[] digits) {
		int carry = 1;
		for (int i = digits.length - 1; i >= 0; i--) {
			int digit = digits[i];
			int sum = digit + carry;
			carry = sum / 10;
			sum = sum % 10;
			digits[i] = sum;
		}
		if (carry == 1) {
			int[] res = new int[digits.length + 1];
			res[0] = 1;
			for (int i = 1; i < res.length; i++) {
				res[i] = digits[i - 1];
			}
			return res;
		}
		return digits;
	}

	public int[] minusOne(int[] digits) {
		int borrow = -1;
		for (int i = digits.length - 1; i >= 0; i--) {
			int digit = digits[i];
			int sum = digit + borrow;
			if (digit == 0 && i > 0) {
				digits[i] = 9;
				borrow = -1;
			} else {
				borrow = 0;
				digits[i] = sum;
			}
		}

		if (digits.length > 1 && digits[0] == 0) {
			int[] res = new int[digits.length - 1];
			for (int i = 0; i < res.length; i++) {
				res[i] = digits[i + 1];
			}
			return res;
		}
		return digits;
	}

	public String subtract(String s1, String s2) {
		char[] a = new StringBuilder(s1).reverse().toString().toCharArray();
		char[] b = new StringBuilder(s2).reverse().toString().toCharArray();
		int len1 = a.length, len2 = b.length;
		int len = len1 > len2 ? len1 : len2;
		int[] res = new int[len];

		boolean sign = false;
		if (len1 < len)
			sign = true;
		else if (len1 == len) {
			int i = len1 - 1;
			while (i > 0 && a[i] == b[i])
				i--;
			if (a[i] < b[i])
				sign = true;
		}

		for (int i = 0; i < len; i++) {
			int aint = i < len1 ? (a[i] - '0') : 0;
			int bint = i < len2 ? (b[i] - '0') : 0;
			if (!sign)
				res[i] = aint - bint;
			else
				res[i] = bint - aint;
		}

		for (int i = 0; i < res.length - 1; i++) {
			if (res[i] < 0) {
				res[i] += 10;
				res[i + 1]--;
			}
		}
		StringBuilder sb = new StringBuilder();
		int k = res.length - 1;
		while (k >= 0 && res[k] == 0)
			k--;
		while (k >= 0) {
			sb.append(res[k--]);
		}
		if (sign)
			return "-" + sb.toString();
		return sb.toString();
	}

	public String multiply(String num1, String num2) {
		int len1 = num1.length(), len2 = num2.length();
		int[] res = new int[len1 + len2];
		for (int i = len1 - 1; i >= 0; i--) {
			int carry = 0;
			int digit1 = num1.charAt(i) - '0';
			for (int j = len2 - 1; j >= 0; j--) {
				int digit2 = num2.charAt(j) - '0';
				int prod = digit1 * digit2 + carry + res[i + j + 1];
				carry = prod / 10;
				prod %= 10;
				res[i + j + 1] = prod;
			}
			res[i] = carry;
		}

		String ans = "";
		int i = 0;
		while (i < res.length - 1 && res[i] == 0) {
			i++;
		}
		while (i < res.length) {
			ans += res[i++];
		}
		// if(ans.isEmpty())
		// return "0";
		return ans;
	}

	public String multiply2(String num1, String num2) {
		int l1 = num1.length(), l2 = num2.length();
		int[] res = new int[l1 + l2];
		for (int i = l1 - 1; i >= 0; i--) {
			int dig1 = num1.charAt(i) - '0';
			for (int j = l2 - 1; j >= 0; j--) {
				int dig2 = num2.charAt(j) - '0';
				int prod = dig1 * dig2;
				res[i + j + 1] += prod;
			}
		}
		String ans = "";
		int carry = 0;
		for (int i = res.length - 1; i >= 0; i--) {
			int sum = res[i] + carry;
			carry = sum / 10;
			sum = sum % 10;
			ans = sum + ans;
		}
		int i = 0;
		while (i < ans.length() && ans.charAt(i) == '0') {
			i++;
		}
		return i == ans.length() ? "0" : ans.substring(i);
	}

	public int calculateBasic(String s) {
		int res = 0;
		Stack<Integer> stk = new Stack<Integer>();
		int sign = 1;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (c >= '0' && c <= '9') {
				int t = c - '0';
				while (i + 1 < s.length() && Character.isDigit(s.charAt(i + 1))) {
					t = t * 10 + (s.charAt(i + 1) - '0');
					i++;
				}
				res += sign * t;
			}
			if (c == '-')
				sign = -1;
			else if (c == '(') {
				stk.push(res);
				res = 0;
				stk.push(sign);
				sign = 1;
			} else if (c == ')') {
				res = res * stk.pop() + stk.pop();
				sign = 1;
			} else {
				sign = 1;
			}
		}
		return res;
	}

	public int calculateBasic1(String s) {
		Stack<Integer> stk = new Stack<Integer>();
		int res = 0, sign = 1;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (Character.isDigit(c)) {
				int sum = c - '0';
				while (i + 1 < s.length() && Character.isDigit(s.charAt(i + 1))) {
					sum = sum * 10 + (s.charAt(i + 1) - '0');
					i++;
				}
				res += sum * sign;
			} else if (c == '+')
				sign = 1;
			else if (c == '-')
				sign = -1;
			else if (c == '(') {
				stk.push(res);
				stk.push(sign);
				res = 0;
				sign = 1;
			} else if (c == ')') {
				res = res * stk.pop() + stk.pop();
			}
		}
		return res;
	}

	public List<List<Integer>> permute(int[] nums) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		boolean[] used = new boolean[nums.length];
		permute(nums, 0, sol, res, used);
		return res;
	}

	public void permute(int[] nums, int cur, List<Integer> sol,
			List<List<Integer>> res, boolean[] used) {
		if (cur == nums.length) {
			res.add(new ArrayList<Integer>(sol));
			return;
		}
		for (int i = 0; i < nums.length; i++) {
			if (!used[i]) {
				sol.add(nums[i]);
				used[i] = true;
				permute(nums, cur + 1, sol, res, used);
				sol.remove(sol.size() - 1);
				used[i] = false;
			}
		}
	}

	public List<List<Integer>> permuteIterative2(int[] nums) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		res.add(new ArrayList<Integer>());

		for (int i = 0; i < nums.length; i++) {
			List<List<Integer>> current = new ArrayList<List<Integer>>();
			for (List<Integer> lst : res) {
				for (int j = 0; j <= i; j++) {
					List<Integer> newSol = new ArrayList<Integer>(lst);
					newSol.add(j, nums[i]);
					current.add(newSol);
				}
			}
			res = current;
		}
		return res;
	}

	public List<List<Integer>> permuteIterative(int[] nums) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		// start from an empty list
		res.add(new ArrayList<Integer>());
		for (int i = 0; i < nums.length; i++) {
			// list of list in current iteration of the array num
			List<List<Integer>> current = new ArrayList<List<Integer>>();
			for (List<Integer> lst : res) {
				// # of locations to insert is largest index + 1
				for (int j = 0; j < lst.size() + 1; j++) {
					lst.add(j, nums[i]);
					current.add(new ArrayList<Integer>(lst));
					// remove num[i] added
					lst.remove(j);
				}
			}
			res = current;
		}
		return res;
	}

	public List<List<Integer>> permuteUnique(int[] nums) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		boolean[] used = new boolean[nums.length];
		Arrays.sort(nums);
		permuteUniqueUtil(nums, 0, sol, res, used);
		return res;
	}

	public void permuteUniqueUtil(int[] nums, int dep, List<Integer> sol,
			List<List<Integer>> res, boolean[] used) {
		if (dep == nums.length) {
			res.add(new ArrayList<Integer>(sol));
			return;
		}
		for (int i = 0; i < nums.length; i++) {
			if (i != 0 && nums[i] == nums[i - 1] && !used[i - 1])
				continue;
			if (!used[i]) {
				sol.add(nums[i]);
				used[i] = true;
				permuteUniqueUtil(nums, dep + 1, sol, res, used);
				sol.remove(sol.size() - 1);
				used[i] = false;
			}
		}
	}

	public List<List<Integer>> permuteUniqueIterative(int[] nums) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		res.add(new ArrayList<Integer>());

		for (int i = 0; i < nums.length; i++) {
			Set<List<Integer>> current = new HashSet<List<Integer>>();
			for (List<Integer> l : res) {
				for (int j = 0; j < l.size() + 1; j++) {
					l.add(j, nums[i]);
					current.add(new ArrayList<Integer>(l));
					l.remove(j);
				}
			}
			res = new ArrayList<List<Integer>>(current);
		}
		return res;
	}

	public List<List<Integer>> combine(int n, int k) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		combine(n, k, 0, sol, res, 1);
		return res;
	}

	public void combine(int n, int k, int dep, List<Integer> sol,
			List<List<Integer>> res, int cur) {
		if (dep == k) {
			res.add(new ArrayList<Integer>(sol));
			return;
		}
		for (int i = cur; i <= n; i++) {
			sol.add(i);
			combine(n, k, dep + 1, sol, res, i + 1);
			sol.remove(sol.size() - 1);
		}
	}

	public void reorderList(ListNode head) {
		if (head == null || head.next == null)
			return;
		ListNode fast = head, slow = head;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			if (fast == null)
				break;
			slow = slow.next;
		}
		ListNode secondHead = slow.next;
		slow.next = null;

		secondHead = reverseList(secondHead);
		ListNode cur1 = head;
		ListNode cur2 = secondHead;

		while (cur1 != null && cur2 != null) {
			ListNode pnext1 = cur1.next;
			ListNode pnext2 = cur2.next;
			cur1.next = cur2;
			cur2.next = pnext1;

			cur1 = pnext1;
			cur2 = pnext2;
		}
	}

	// public ListNode reverseList(ListNode head){
	// if(head==null||head.next==null)
	// return head;
	// ListNode dummy=new ListNode(0);
	// dummy.next=head;
	// ListNode last=head, pre=dummy, cur=head.next;
	// while(cur!=null){
	// last.next=cur.next;
	// cur.next=pre.next;
	// pre.next=cur;
	//
	// cur=last.next;
	// }
	// return dummy.next;
	// }

	public TreeNode sortedListToBST(ListNode head) {
		if (head == null)
			return null;
		int len = 0;
		ListNode cur = head;
		while (cur != null) {
			len++;
			cur = cur.next;
		}
		return sortedListToBST(head, 0, len);
	}

	public TreeNode sortedListToBST(ListNode head, int left, int right) {
		if (head == null || left > right)
			return null;
		int mid = left + (right - left) / 2;
		ListNode cur = head;
		for (int i = left; i < mid; i++) {
			cur = cur.next;
		}
		TreeNode root = new TreeNode(cur.val);
		root.left = sortedListToBST(head, left, mid - 1);
		root.right = sortedListToBST(cur.next, mid + 1, right);
		return root;

	}

	public TreeNode sortedListToBST2(ListNode head) {
		if (head == null)
			return null;
		ListNode fast = head, slow = head, pre = null;
		while (fast != null && fast.next != null) {
			pre = slow;
			slow = slow.next;
			fast = fast.next.next;
		}
		ListNode second = slow.next;
		slow.next = null;
		TreeNode root = new TreeNode(slow.val);
		root.right = sortedListToBST(second);
		if (pre != null) {
			pre.next = null;
			root.left = sortedListToBST(head);
		} else {
			return root;
		}
		return root;
	}

	public TreeNode sortedArrayToBST(int[] nums) {
		return sortedArrayToBST(nums, 0, nums.length - 1);
	}

	public TreeNode sortedArrayToBST(int[] nums, int left, int right) {
		if (left > right)
			return null;
		int mid = left + (right - left) / 2;
		TreeNode root = new TreeNode(nums[mid]);

		root.left = sortedArrayToBST(nums, left, mid - 1);
		root.right = sortedArrayToBST(nums, mid + 1, right);
		return root;
	}

	public TreeNode buildTree(int[] preorder, int[] inorder) {
		return buildTree(preorder, 0, preorder.length - 1, inorder, 0,
				inorder.length - 1);
	}

	public TreeNode buildTree(int[] preorder, int beg1, int end1,
			int[] inorder, int beg2, int end2) {
		if (beg1 > end1)
			return null;
		TreeNode root = new TreeNode(preorder[beg1]);
		int index = -1;
		for (int i = beg2; i <= end2; i++) {
			if (inorder[i] == root.val) {
				index = i;
				break;
			}
		}
		int length = index - beg2;
		root.left = buildTree(preorder, beg1 + 1, beg1 + length, inorder, beg2,
				index - 1);
		root.right = buildTree(preorder, beg1 + index - beg2 + 1, end1,
				inorder, index + 1, end2);
		return root;
	}

	public TreeNode buildTree2(int[] inorder, int[] postorder) {
		return buildTreeUtil(inorder, 0, inorder.length - 1, postorder, 0,
				postorder.length - 1);
	}

	public TreeNode buildTreeUtil(int[] inorder, int beg1, int end1,
			int[] postorder, int beg2, int end2) {
		if (beg1 > end1)
			return null;
		TreeNode root = new TreeNode(postorder[end2]);
		int index = -1;
		for (int i = beg1; i <= end1; i++) {
			if (inorder[i] == root.val) {
				index = i;
				break;
			}
		}
		int len = index - beg1;
		root.left = buildTreeUtil(inorder, beg1, index - 1, postorder, beg2,
				beg2 + len - 1);
		root.right = buildTreeUtil(inorder, index + 1, end1, postorder, beg2
				+ len, end2 - 1);
		return root;
	}

	public String serialize(TreeNode root) {
		StringBuilder sb = new StringBuilder();
		serialize(root, sb);
		return sb.toString();
	}

	public void serialize(TreeNode root, StringBuilder sb) {
		if (root == null) {
			sb.append("#,");
			return;
		}
		sb.append(root.val + ",");
		serialize(root.left, sb);
		serialize(root.right, sb);
	}

	// Decodes your encoded data to tree.
	public TreeNode deserialize(String data) {
		String[] strs = data.split(",");
		Deque<String> nodes = new LinkedList<String>();
		nodes.addAll(Arrays.asList(strs));
		return buildTree(nodes);
	}

	public TreeNode buildTree(Deque<String> nodes) {
		String val = nodes.remove();
		if (val.equals("#"))
			return null;
		TreeNode root = new TreeNode(Integer.parseInt(val));
		root.left = buildTree(nodes);
		root.right = buildTree(nodes);
		return root;
	}

	public String serialize2(TreeNode root) {
		StringBuilder sb = new StringBuilder();
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		que.add(root);
		while (!que.isEmpty()) {
			TreeNode node = que.poll();
			if (node == null) {
				sb.append("null,");
			} else {
				sb.append(node.val + ",");
				que.offer(node.left);
				que.offer(node.right);
			}
		}
		return sb.toString();
	}

	// Decodes your encoded data to tree.
	public TreeNode deserialize2(String data) {
		String[] strs = data.split(",");
		if (strs.length == 1)
			return null;
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		TreeNode root = new TreeNode(Integer.parseInt(strs[0]));
		que.offer(root);
		int i = 1;
		while (i < strs.length) {
			TreeNode node = que.poll();
			String left = strs[i++], right = strs[i++];
			if (!left.equals("null")) {
				TreeNode leftNode = new TreeNode(Integer.parseInt(left));
				node.left = leftNode;
				que.offer(leftNode);
			}
			if (!right.equals("null")) {
				TreeNode rightNode = new TreeNode(Integer.parseInt(right));
				node.right = rightNode;
				que.offer(rightNode);
			}
		}
		return root;
	}

	public int missingNumber(int[] nums) {
		// for(int i=0;i<nums.length;i++){
		// while(nums[i]!=i){
		// if(nums[i]<0||nums[i]>=nums.length||nums[i]==nums[nums[i]])
		// break;
		// int t=nums[i];
		// nums[i]=nums[t];
		// nums[t]=t;
		// }
		// }
		// for(int i=0;i<nums.length;i++){
		// if(nums[i]!=i)
		// return i;
		// }
		// return nums.length;
		int n = nums.length;
		int sum = 0;
		for (int i = 0; i < n; i++) {
			sum += nums[i];
		}
		return n * (n + 1) / 2 - sum;
	}

	// visited = -1, means this node has been visited.
	// visited = 1, means this node has been validated which does not include a
	// circle.
	// Thus if we saw that a node has been validated,
	// we don't need to calculate again to find out the circle starting from
	// this node.
	// e.g. [0, 1] [1, 2] [2, 3] [3, 4]. For the node 0, we have already
	// validated 2 3 and 4 do not have a circle.
	// Thus we don't need to calculate for the node 2 3 4 again.

	public boolean canFinish(int numCourses, int[][] prerequisites) {
		HashMap<Integer, List<Integer>> map = new HashMap<Integer, List<Integer>>();
		for (int i = 0; i < prerequisites.length; i++) {
			int pre = prerequisites[i][1];
			int cur = prerequisites[i][0];
			if (!map.containsKey(cur)) {
				List<Integer> pres = new ArrayList<Integer>();
				pres.add(pre);
				map.put(cur, pres);
			} else {
				map.get(cur).add(pre);
			}
		}
		int[] visited = new int[numCourses];
		for (int i = 0; i < numCourses; i++) {
			if (dfsDetectCycle(i, visited, map))
				return false;
		}
		return true;
	}

	public boolean dfsDetectCycle(int course, int[] visited,
			HashMap<Integer, List<Integer>> map) {
		if (visited[course] == -1)
			return true;
		if (visited[course] == 1)
			return false;
		visited[course] = -1;
		if (map.containsKey(course)) {
			for (int i : map.get(course)) {
				if (dfsDetectCycle(i, visited, map))
					return true;
			}
		}
		visited[course] = 1;
		return false;
	}

	public boolean canFinish2(int numCourses, int[][] prerequisites) {
		int len = prerequisites.length;
		int[] count = new int[numCourses];
		// counter for number of prerequisites
		for (int i = 0; i < len; i++) {
			count[prerequisites[i][0]]++;
		}
		// store courses that have no prerequisites
		Queue<Integer> que = new LinkedList<Integer>();
		for (int i = 0; i < numCourses; i++) {
			if (count[i] == 0)
				que.add(i);
		}

		int canFinish = que.size();
		while (!que.isEmpty()) {
			int course = que.remove();
			for (int i = 0; i < len; i++) {
				if (prerequisites[i][1] == course) {
					count[prerequisites[i][0]]--;
					if (count[prerequisites[i][0]] == 0) {
						canFinish++;
						que.add(prerequisites[i][0]);
					}
				}
			}
		}
		return canFinish == numCourses;
	}

	public void binaryTree2DoublyLinkedList(TreeNode root) {
		if (root == null) {
			return;
		}
		binaryTree2DoublyLinkedListUtil(root);
		connectHeadAndTail(root);
	}

	public void connectHeadAndTail(TreeNode root) {
		TreeNode head = root;
		while (head.left != null) {
			head = head.left;
		}
		TreeNode tail = root;
		while (tail.right != null) {
			tail = tail.right;
		}
		head.left = tail;
		tail.right = head;
	}

	public void binaryTree2DoublyLinkedListUtil(TreeNode root) {
		if (root == null)
			return;
		if (root.left != null) {
			TreeNode left = root.left;
			binaryTree2DoublyLinkedListUtil(left);
			while (left.right != null) {
				left = left.right;
			}
			left.right = root;
			root.left = left;
		}
		if (root.right != null) {
			TreeNode right = root.right;
			binaryTree2DoublyLinkedListUtil(right);
			while (right.left != null) {
				right = right.left;
			}
			right.left = root;
			root.right = right;
		}
	}

	public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
		if (t < 0)
			return false;
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		t += 1;
		for (int i = 0; i < nums.length; i++) {
			if (i > k) {
				map.remove(getID(nums[i - k - 1], t));
			}
			int m = getID(nums[i], t);
			if (map.containsKey(m))
				return true;
			if (map.containsKey(m - 1)
					&& Math.abs(nums[i] - map.get(m - 1)) < t)
				return true;
			if (map.containsKey(m + 1)
					&& Math.abs(nums[i] - map.get(m + 1)) < t)
				return true;
			map.put(m, nums[i]);
		}
		return false;
	}

	public int getID(int i, int t) {
		return i < 0 ? (i + 1) / t - 1 : i / t;
	}

	public boolean containsNearbyAlmostDuplicate2(int[] nums, int k, int t) {
		if (k < 1 || t < 0)
			return false;
		Map<Long, Long> map = new HashMap<>();
		for (int i = 0; i < nums.length; i++) {
			long remappedNum = (long) nums[i] - Integer.MIN_VALUE;
			long bucket = remappedNum / ((long) t + 1);
			if (map.containsKey(bucket)) {
				return true;
			}
			if (map.containsKey(bucket - 1)
					&& Math.abs(remappedNum - map.get(bucket - 1)) <= t) {

				return true;
			}
			if (map.containsKey(bucket + 1)
					&& Math.abs(remappedNum - map.get(bucket + 1)) <= t) {
				return true;
			}
			map.put(bucket, remappedNum);

			if (i >= k) {
				long lastBucket = ((long) nums[i - k] - Integer.MIN_VALUE)
						/ ((long) t + 1);
				map.remove(lastBucket);
			}

		}
		return false;

		// for (int i = 0; i < nums.length; i++) {
		// long remappedNum = (long) nums[i] - Integer.MIN_VALUE;
		// long bucket = remappedNum / ((long) t + 1);
		// if (map.containsKey(bucket)
		// || (map.containsKey(bucket - 1) && remappedNum - map.get(bucket - 1)
		// <= t)
		// || (map.containsKey(bucket + 1) && map.get(bucket + 1) - remappedNum
		// <= t))
		// return true;
		// if (map.entrySet().size() >= k) {
		// long lastBucket = ((long) nums[i - k] - Integer.MIN_VALUE) / ((long)
		// t + 1);
		// map.remove(lastBucket);
		// }
		// map.put(bucket, remappedNum);
		// }
		// return false;
	}

	public List<Integer> countSmaller(int[] nums) {
		List<Integer> res = new ArrayList<Integer>();
		if (nums.length == 0)
			return res;
		TreeNode root = null;

		for (int i = nums.length - 1; i >= 0; i--) {
			root = TreeNode.insert(root, nums[i]);
			int rank = TreeNode.rank(root, nums[i]);
			res.add(0, rank);
		}
		return res;
	}

	public class TreeNode2 {
		int val;
		TreeNode2 left;
		TreeNode2 right;
		int selfSize;
		int leftSize;

		public TreeNode2(int val) {
			this.val = val;
			selfSize = 1;
		}
	}

	public List<Integer> countSmaller2(int[] nums) {
		List<Integer> res = new ArrayList<Integer>();
		int n = nums.length;
		if (n == 0)
			return res;
		TreeNode2 root = new TreeNode2(nums[n - 1]);
		res.add(0);
		for (int i = nums.length - 2; i >= 0; i--) {
			res.add(0, addToTree(root, nums[i]));
		}
		return res;
	}

	public List<Integer> countSmallerBefore(int[] nums) {
		List<Integer> res = new ArrayList<Integer>();
		int n = nums.length;
		if (n == 0)
			return res;
		TreeNode2 root = new TreeNode2(nums[0]);
		res.add(0);
		for (int i = 1; i < n; i++) {
			res.add(addToTree(root, nums[i]));
		}
		return res;
	}

	public int addToTree(TreeNode2 root, int val) {
		int num = 0;
		TreeNode2 cur = root;
		while (true) {
			if (cur.val < val) {
				num += cur.leftSize + cur.selfSize;
				if (cur.right == null) {
					cur.right = new TreeNode2(val);
					break;
				}
				cur = cur.right;
			} else if (cur.val > val) {
				cur.leftSize++;
				if (cur.left == null) {
					cur.left = new TreeNode2(val);
					break;
				}
				cur = cur.left;
			} else {
				cur.selfSize++;
				num += cur.leftSize;
				break;
			}
		}
		return num;
	}

	public RandomListNode copyRandomList(RandomListNode head) {
		if (head == null)
			return null;
		RandomListNode cur = head;
		while (cur != null) {
			RandomListNode copy = new RandomListNode(cur.label);
			copy.next = cur.next;
			cur.next = copy;
			cur = copy.next;
		}
		cur = head;
		while (cur != null) {
			if (cur.random != null) {
				cur.next.random = cur.random.next;
			}
			cur = cur.next.next;
		}

		cur = head;
		RandomListNode copyHead = head.next;
		RandomListNode cur1 = copyHead;
		while (cur != null) {
			cur.next = cur.next.next;
			if (cur1.next != null)
				cur1.next = cur1.next.next;
			cur = cur.next;
			cur1 = cur1.next;
		}
		return copyHead;
	}

	public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
		if (node == null)
			return null;
		HashMap<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<UndirectedGraphNode, UndirectedGraphNode>();
		UndirectedGraphNode copy = new UndirectedGraphNode(node.label);
		map.put(node, copy);
		Queue<UndirectedGraphNode> que = new LinkedList<UndirectedGraphNode>();
		que.add(node);

		while (!que.isEmpty()) {
			UndirectedGraphNode cur = que.remove();
			List<UndirectedGraphNode> neighbors = cur.neighbors;
			for (int i = 0; i < neighbors.size(); i++) {
				UndirectedGraphNode neighbor = neighbors.get(i);
				if (!map.containsKey(neighbor)) {
					UndirectedGraphNode clone = new UndirectedGraphNode(
							neighbor.label);
					map.put(neighbor, clone);
					que.add(neighbor);
					map.get(cur).neighbors.add(clone);
				} else {
					map.get(cur).neighbors.add(map.get(neighbor));
				}
			}
		}
		return copy;
	}

	public UndirectedGraphNode cloneGraphDFS(UndirectedGraphNode node) {
		if (node == null)
			return null;
		HashMap<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<UndirectedGraphNode, UndirectedGraphNode>();
		UndirectedGraphNode copy = new UndirectedGraphNode(node.label);
		map.put(node, copy);
		dfsClone(node, map);
		return copy;
	}

	public void dfsClone(UndirectedGraphNode node,
			HashMap<UndirectedGraphNode, UndirectedGraphNode> map) {
		if (node == null)
			return;
		List<UndirectedGraphNode> neighbors = node.neighbors;
		for (UndirectedGraphNode neighbor : neighbors) {
			if (!map.containsKey(neighbor)) {
				UndirectedGraphNode clone = new UndirectedGraphNode(
						neighbor.label);
				map.put(neighbor, clone);
				dfsClone(neighbor, map);
			}
			map.get(node).neighbors.add(map.get(neighbor));
		}
	}

	public int lengthOfLongestSubstring(String s) {
		if (s.length() < 2)
			return s.length();
		int maxLen = 0;
		int left = 0;
		Set<Character> set = new HashSet<Character>();
		for (int right = 0; right < s.length(); right++) {
			char c = s.charAt(right);
			if (!set.contains(c))
				set.add(c);
			else {
				maxLen = Math.max(right - left, maxLen);
				while (s.charAt(left) != s.charAt(right)) {
					set.remove(s.charAt(left));
					left++;
				}
				left++;
			}
		}
		maxLen = Math.max(maxLen, s.length() - left);
		return maxLen;
	}

	public int lengthOfLongestSubstring2(String s) {
		Map<Character, Integer> map = new HashMap<Character, Integer>();
		int max = 0, start = 0;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (map.containsKey(c)) {
				max = Math.max(max, i - start);
				int dup = map.get(c);
				for (int j = start; j <= dup; j++) {
					map.remove(s.charAt(j));
				}
				start = dup + 1;
			}
			map.put(c, i);
		}
		max = Math.max(max, s.length() - start);
		return max;
	}

	public int lengthOfLongestSubstring3(String s) {
		Map<Character, Integer> map = new HashMap<Character, Integer>();
		int max = 0, start = 0;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (map.containsKey(c) && start <= map.get(c)) {
				start = map.get(c) + 1;
			} else {
				max = Math.max(max, i - start + 1);
			}
			map.put(c, i);
		}

		return max;
	}

	public int lengthOfLongestSubstring4(String s) {
		Set<Character> set = new HashSet<Character>();
		int max = 0, i = 0, j = 0;
		while (i < s.length()) {
			if (!set.contains(s.charAt(i))) {
				set.add(s.charAt(i++));
				max = Math.max(max, set.size());
			} else {
				set.remove(s.charAt(j++));
			}
		}

		return max;
	}

	public int lengthOfLongestSubstringTwoDistinct(String s) {
		if (s.length() < 3)
			return s.length();
		int maxLen = 2;
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		int hi = 0, lo = 0;
		while (hi < s.length()) {
			if (map.size() <= 2) {
				char c = s.charAt(hi);
				map.put(c, hi);
				hi++;
			}
			if (map.size() > 2) {
				int leftmost = s.length();
				for (int i : map.values()) {
					leftmost = Math.min(leftmost, i);
				}
				char c = s.charAt(leftmost);
				map.remove(c);
				lo = leftmost + 1;
			}
			maxLen = Math.max(maxLen, hi - lo);
		}
		return maxLen;
	}

	public String maxSubStringKUniqueChars(String s, int k) {
		// declare a counter
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		int start = 0;
		int maxLen = 0;
		String maxSubstring = "";

		for (int i = 0; i < s.length(); i++) {
			// add each char to the counter
			char c = s.charAt(i);
			if (map.containsKey(c)) {
				map.put(c, map.get(c) + 1);
			} else {
				map.put(c, 1);
			}

			if (map.size() == k + 1) {
				// get maximum
				int range = i - start;
				if (range > maxLen) {
					maxLen = range;
					maxSubstring = s.substring(start, i);
				}

				// move left cursor toward right, so that substring contains
				// only k chars
				while (map.size() > k) {
					char first = s.charAt(start);
					int count = map.get(first);
					if (count > 1) {
						map.put(first, count - 1);
					} else {
						map.remove(first);
					}
					start++;
				}
			}
		}

		if (map.size() == k && maxLen == 0) {
			return s;
		}

		return maxSubstring;
	}

	public boolean wordBreak(String s, Set<String> wordDict) {
		// if(s.length()==0)
		// return true;
		// for(int i=1;i<=s.length();i++){
		// String word=s.substring(0, i);
		// if(wordDict.contains(word)&&wordBreak(s.substring(i), wordDict)){
		// return true;
		// }
		// }
		// return false;
		int n = s.length();
		boolean[] dp = new boolean[n + 1];
		dp[0] = true;
		for (int i = 1; i <= s.length(); i++) {
			for (int j = 0; j < i; j++) {
				String word = s.substring(j, i);
				if (dp[j] && wordDict.contains(word))
					dp[i] = true;
			}
		}
		return dp[n];
	}

	public List<String> wordBreak2DFS(String s, Set<String> wordDict) {
		List<String> res = new ArrayList<String>();
		wordBreakUtil(s, wordDict, res, "");
		return res;
	}

	public void wordBreakUtil(String s, Set<String> wordDict, List<String> res,
			String sol) {
		if (s.length() == 0) {
			res.add(sol.trim());
			return;
		}
		for (int i = 1; i <= s.length(); i++) {
			String word = s.substring(0, i);
			if (wordDict.contains(word)) {
				wordBreakUtil(s.substring(i), wordDict, res, sol + word + " ");
			}
		}
	}

	public List<String> wordBreakMemorize2(String s, Set<String> wordDict) {
		HashMap<String, List<String>> map = new HashMap<String, List<String>>();
		return wordBreakMemorize2Util(s, wordDict, map);
	}

	public List<String> wordBreakMemorize2Util(String s, Set<String> wordDict,
			Map<String, List<String>> map) {
		if (map.containsKey(s))
			return map.get(s);
		List<String> res = new ArrayList<String>();
		for (int i = 1; i <= s.length(); i++) {
			String word = s.substring(0, i);
			String rem = s.substring(i);
			if (wordDict.contains(word)) {
				List<String> temp = wordBreakMemorize2Util(rem, wordDict, map);
				for (String words : temp) {
					res.add(word + " " + words);
				}
				if (rem.isEmpty())
					res.add(word);
			}
		}
		map.put(s, res);
		return res;
	}

	public List<String> wordBreakMemorize(String s, Set<String> wordDict) {
		HashMap<String, List<String>> map = new HashMap<String, List<String>>();
		return wordBreakUtil(s, wordDict, map);
	}

	public List<String> wordBreakUtil(String s, Set<String> wordDict,
			HashMap<String, List<String>> map) {
		if (map.containsKey(s)) {
			return map.get(s);
		}
		List<String> res = new ArrayList<String>();
		if (s.length() == 0) {
			res.add("");
			return res;
		}
		for (String word : wordDict) {
			if (s.startsWith(word)) {
				List<String> sublist = wordBreakUtil(
						s.substring(word.length()), wordDict, map);
				for (String str : sublist) {
					String sol = word + (str.isEmpty() ? "" : " ") + str;
					res.add(sol);
				}
			}
		}
		map.put(s, res);
		return res;
	}

	public List<String> wordBreak2(String s, Set<String> wordDict) {
		boolean[] dp = new boolean[s.length() + 1];
		dp[0] = true;
		for (int i = 1; i <= s.length(); i++) {
			String word = s.substring(0, i);
			if (wordDict.contains(word))
				dp[i] = true;
			else {
				for (int j = 1; j < i; j++) {
					if (dp[j] && wordDict.contains(s.substring(j, i))) {
						dp[i] = true;
						break;
					}
				}
			}
		}
		if (!dp[s.length()])
			return new ArrayList<String>();
		return wordBreak(s, wordDict, dp, s.length());
	}

	public List<String> wordBreak(String s, Set<String> wordDict, boolean[] dp,
			int end) {
		List<String> res = new ArrayList<String>();
		for (int i = 0; i <= end; i++) {
			String word = s.substring(i, end);
			if (dp[i] && wordDict.contains(word)) {
				List<String> subsol = wordBreak(s, wordDict, dp, i);
				for (String t : subsol) {
					res.add(t + " " + word);
				}
			}
		}
		if (wordDict.contains(s.substring(0, end)))
			res.add(s.substring(0, end));
		return res;
	}

	public List<String> wordBreak3(String s, Set<String> wordDict) {
		List<String> res = new ArrayList<String>();
		int n = s.length();
		boolean[][] dp = new boolean[n][n + 1];
		for (int i = n - 1; i >= 0; i--) {
			for (int j = i + 1; j <= n; j++) {
				String word = s.substring(i, j);
				if (wordDict.contains(word) && j == n) {
					dp[i][j - 1] = true;
					dp[i][n] = true;
				} else if (wordDict.contains(word) && j < n && dp[j][n]) {
					dp[i][j - 1] = true;
					dp[i][n] = true;
				}
			}
		}
		if (!dp[0][n])
			return res;
		dfsWordBreak(s, wordDict, dp, res, "", 0);
		return res;
	}

	public void dfsWordBreak(String s, Set<String> wordDict, boolean[][] dp,
			List<String> res, String sol, int cur) {
		if (cur == s.length()) {
			res.add(sol);
			return;
		}

		for (int i = cur; i < s.length(); i++) {
			if (dp[cur][i]) {
				String word = s.substring(cur, i + 1);
				String sub = "";
				if (i < s.length() - 1)
					sub = sol + word + " ";
				else
					sub = sol + word;
				dfsWordBreak(s, wordDict, dp, res, sub, i + 1);
			}
		}
	}

	public int climbStairs(int n) {
		if (n < 3)
			return n;
		int first = 1, second = 2;
		for (int i = 3; i <= n; i++) {
			int total = first + second;
			first = second;
			second = total;
		}
		return second;
	}

	public void merge(int[] nums1, int m, int[] nums2, int n) {
		int len = m + n - 1;
		int i = m - 1, j = n - 1;
		while (i >= 0 && j >= 0) {
			if (nums1[i] > nums2[j])
				nums1[len--] = nums1[i--];
			else
				nums1[len--] = nums2[j--];
		}
		while (j >= 0) {
			nums1[len--] = nums2[j--];
		}
	}

	public List<String> findRepeatedDnaSequences(String s) {
		List<String> res = new ArrayList<String>();
		if (s.length() < 10)
			return res;
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		for (int i = 0; i < s.length() - 9; i++) {
			String substring = s.substring(i, i + 10);
			if (map.containsKey(substring)) {
				if (map.get(substring) == 1)
					res.add(substring);
				map.put(substring, 2);
			} else {
				map.put(substring, 1);
			}
		}
		return res;
	}

	/*
	 * 实际上我们的哈希表可以不用存整个子串，因为我们知道子串只有10位，且每一位只可能有4种不同的字母，
	 * 那我们可以用4^10个数字来表示每种不同的序列，因为4^10=2^20<2^32所以我们可以用一个Integer来表示。
	 * 具体的编码方法是用每两位bit表示一个字符。
	 */

	public List<String> findRepeatedDnaSequences2(String s) {
		List<String> res = new ArrayList<String>();
		if (s.length() < 10)
			return res;
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < s.length() - 9; i++) {
			String substring = s.substring(i, i + 10);
			int code = encode(substring);
			if (map.containsKey(code)) {
				if (map.get(code) == 1)
					res.add(substring);
				map.put(code, 2);
			} else {
				map.put(code, 1);
			}
		}
		return res;
	}

	public int encode(String s) {
		int code = 0;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			code <<= 2;
			switch (c) {
			case 'A':
				code += 0;
				break;
			case 'C':
				code += 1;
				break;
			case 'T':
				code += 2;
				break;
			case 'G':
				code += 3;
				break;

			}
		}
		return code;
	}

	public List<String> findRepeatedDnaSequences3(String s) {
		Set<String> res = new HashSet<String>();
		Set<String> seen = new HashSet<String>();

		for (int i = 0; i < s.length() - 9; i++) {
			String seq = s.substring(i, i + 10);
			if (!seen.add(seq))
				res.add(seq);
		}
		return new ArrayList<String>(res);
	}

	public List<Integer> findSubstring(String s, String[] words) {
		int n = words.length;
		int len = words[0].length();
		List<Integer> res = new ArrayList<Integer>();
		if (s.length() < n * len)
			return res;
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		for (String word : words) {
			if (map.containsKey(word)) {
				map.put(word, map.get(word) + 1);
			} else {
				map.put(word, 1);
			}
		}

		for (int i = 0; i <= s.length() - n * len; i++) {
			HashMap<String, Integer> found = new HashMap<String, Integer>();
			int j = 0;
			for (; j < n; j++) {
				String sub = s.substring(i + j * len, i + j * len + len);
				if (!map.containsKey(sub))
					break;
				if (found.containsKey(sub)) {
					found.put(sub, found.get(sub) + 1);
					if (found.get(sub) > map.get(sub))
						break;
				} else {
					found.put(sub, 1);
				}
			}
			if (j == n)
				res.add(i);
		}
		return res;
	}

	// by removing the found word to improve the complexty
	public List<Integer> findSubstring2(String s, String[] words) {
		int n = words.length;
		int len = words[0].length();
		List<Integer> res = new ArrayList<Integer>();
		if (s.length() < n * len)
			return res;
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		for (String word : words) {
			if (map.containsKey(word)) {
				map.put(word, map.get(word) + 1);
			} else {
				map.put(word, 1);
			}
		}

		for (int i = 0; i <= s.length() - n * len; i++) {
			HashMap<String, Integer> found = new HashMap<String, Integer>(map);
			for (int j = 0; j < n; j++) {
				String sub = s.substring(i + j * len, i + (j + 1) * len);
				if (!found.containsKey(sub))
					break;
				found.put(sub, found.get(sub) - 1);
				if (found.get(sub) == 0)
					found.remove(sub);
				if (found.isEmpty())
					res.add(i);
			}
		}
		return res;
	}

	public String minWindow(String s, String t) {
		if (s.length() < t.length())
			return "";
		int[] needFind = new int[256];
		for (int i = 0; i < t.length(); i++) {
			needFind[t.charAt(i)]++;
		}
		int[] hasFound = new int[256];
		int count = t.length();
		int minLen = s.length() + 1;
		int start = 0;
		int windowStart = 0;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (needFind[c] == 0)
				continue;
			hasFound[c]++;
			if (hasFound[c] <= needFind[c])
				count--;
			if (count == 0) {
				while (needFind[s.charAt(start)] == 0
						|| hasFound[s.charAt(start)] > needFind[s.charAt(start)]) {
					if (hasFound[s.charAt(start)] > needFind[s.charAt(start)]) {
						hasFound[s.charAt(start)]--;
					}
					start++;
				}
				if (minLen > i - start + 1) {
					minLen = i - start + 1;
					windowStart = start;
				}
			}
		}
		if (count == 0)
			return s.substring(windowStart, windowStart + minLen);
		return "";
	}

	public String minWindow2(String s, String t) {
		if (s.length() < t.length())
			return "";
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		for (int i = 0; i < t.length(); i++) {
			char c = t.charAt(i);
			if (map.containsKey(c))
				map.put(c, map.get(c) + 1);
			else
				map.put(c, 1);
		}
		int windowStart = 0;
		int start = 0;
		int minLen = s.length() + 1;
		int count = 0;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (map.containsKey(c)) {
				map.put(c, map.get(c) - 1);
				if (map.get(c) >= 0)
					count++;
				while (count == t.length()) {
					if (i - start + 1 < minLen) {
						minLen = i - start + 1;
						windowStart = start;
					}
					if (map.containsKey(s.charAt(start))) {
						map.put(s.charAt(start), map.get(s.charAt(start)) + 1);
						if (map.get(s.charAt(start)) > 0)
							count--;
					}
					start++;
				}
			}
		}
		if (minLen > s.length())
			return "";
		return s.substring(windowStart, windowStart + minLen);
	}

	public int minSubArrayLen(int s, int[] nums) {
		int minLen = nums.length + 1;
		int sum = 0;
		int left = 0, right = 0;
		while (right < nums.length) {
			sum += nums[right];
			while (sum >= s) {
				minLen = Math.min(minLen, right - left + 1);
				sum -= nums[left++];
			}
			right++;
		}
		return minLen == nums.length + 1 ? 0 : minLen;
	}

	public int maxProduct(String[] words) {
		int max = 0;
		for (int i = 0; i < words.length; i++) {
			int[] counts = new int[26];
			String word = words[i];
			for (char c : word.toCharArray()) {
				counts[c - 'a']++;
			}
			for (int j = i + 1; j < words.length; j++) {
				String s = words[j];
				int k = 0;
				for (; k < s.length(); k++) {
					if (counts[s.charAt(k) - 'a'] != 0)
						break;
				}
				if (k == s.length()) {
					max = Math.max(max, word.length() * s.length());
				}
			}
		}
		return max;
	}

	public int maxProduct2(String[] words) {
		Arrays.sort(words, new Comparator<String>() {

			@Override
			public int compare(String o1, String o2) {
				// TODO Auto-generated method stub
				return o2.length() - o1.length();
			}

		});

		int max = 0;
		int[][] info = new int[words.length][2];
		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			info[i][0] = getMask(word);
			info[i][1] = word.length();
		}

		for (int i = 0; i < words.length; i++) {
			for (int j = i + 1; j < words.length; j++) {
				if ((info[i][0] & info[j][0]) == 0) {
					max = Math.max(max, info[i][1] * info[j][1]);
					break;
				}
			}
		}
		return max;
	}

	public int getMask(String word) {
		int mask = 0;
		for (int i = 0; i < word.length(); i++) {
			mask |= 1 << (word.charAt(i) - 'a');
		}
		return mask;
	}

	public int maxProduct3(String[] words) {
		int max = 0;
		Arrays.sort(words, new Comparator<String>() {
			@Override
			public int compare(String s1, String s2) {
				return s2.length() - s1.length();
			}
		});

		int[] mask = new int[words.length];
		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			for (int j = 0; j < word.length(); j++) {
				mask[i] |= 1 << (word.charAt(j) - 'a');
			}
		}

		for (int i = 0; i < mask.length; i++) {
			for (int j = i + 1; j < mask.length; j++) {
				if ((mask[i] & mask[j]) == 0) {
					max = Math.max(max, words[i].length() * words[j].length());
					break;
				}
			}
		}
		return max;
	}

	public boolean isIsomorphic(String s, String t) {
		if (s.length() != t.length())
			return false;
		HashMap<Character, Character> map1 = new HashMap<Character, Character>();
		HashMap<Character, Character> map2 = new HashMap<Character, Character>();
		for (int i = 0; i < s.length(); i++) {
			char c1 = s.charAt(i);
			char c2 = t.charAt(i);
			if (!map1.containsKey(c1))
				map1.put(c1, c2);
			else if (map1.get(c1) != c2)
				return false;
			if (!map2.containsKey(c2))
				map2.put(c2, c1);
			else if (map2.get(c2) != c1)
				return false;
		}
		return true;
	}

	public boolean isIsomorphic2(String s, String t) {
		if (s.length() != t.length())
			return false;
		int[] map1 = new int[256];
		int[] map2 = new int[256];

		for (int i = 0; i < s.length(); i++) {
			char c1 = s.charAt(i);
			char c2 = t.charAt(i);

			if (map1[c1] == 0 && map2[c2] == 0) {
				map1[c1] = c2;
				map2[c2] = c1;
			}
			if (map1[c1] != c2 || map2[c2] != c1)
				return false;
		}
		return true;
	}

	public boolean wordPattern(String pattern, String str) {
		String[] strs = str.split(" ");
		if (pattern.length() != strs.length)
			return false;
		Map<Character, String> map1 = new HashMap<Character, String>();
		Map<String, Character> map2 = new HashMap<String, Character>();

		for (int i = 0; i < pattern.length(); i++) {
			char c = pattern.charAt(i);
			String word = strs[i];
			if (!map1.containsKey(c)) {
				map1.put(c, word);
			} else if (!map1.get(c).equals(word)) {
				return false;
			}

			if (!map2.containsKey(word)) {
				map2.put(word, c);
			} else if (map2.get(word) != c) {
				return false;
			}
		}
		return true;
	}

	public boolean wordPatternI_2(String pattern, String str) {
		String[] strs = str.split(" ");
		if (pattern.length() != strs.length)
			return false;

		Map<Character, String> map = new HashMap<Character, String>();
		for (int i = 0; i < pattern.length(); i++) {
			char c = pattern.charAt(i);
			String word = strs[i];
			if (map.containsKey(c) && !map.get(c).equals(word))
				return false;
			else if (!map.containsKey(c) && map.containsValue(word))
				return false;
			map.put(c, word);
		}
		return true;
	}

	public int numIslands(char[][] grid) {
		int m = grid.length;
		if (m == 0)
			return 0;
		int n = grid[0].length;
		int count = 0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (grid[i][j] == '1') {
					dfsIslands(grid, i, j);
					count++;
				}
			}
		}
		return count;
	}

	public void dfsIslands(char[][] grid, int i, int j) {
		if (i < 0 || i >= grid.length || j < 0 || j >= grid[0].length
				|| grid[i][j] != '1')
			return;
		grid[i][j] = '#';
		dfsIslands(grid, i + 1, j);
		dfsIslands(grid, i - 1, j);
		dfsIslands(grid, i, j + 1);
		dfsIslands(grid, i, j - 1);
	}

	public List<Integer> numIslands2(int m, int n, int[][] positions) {
		List<Integer> res = new ArrayList<Integer>();
		int[] ids = new int[m * n];
		Arrays.fill(ids, -1);
		int[][] dirs = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
		int count = 0;
		for (int i = 0; i < positions.length; i++) {
			count++;
			int index = positions[i][0] * n + positions[i][1];
			ids[index] = index;

			for (int j = 0; j < dirs.length; j++) {
				int x = positions[i][0] + dirs[j][0];
				int y = positions[i][1] + dirs[j][1];
				if (x >= 0 && x < m && y >= 0 && y < n && ids[x * n + y] != -1) {
					int root = findRoot(ids, x * n + y);
					if (root != index) {
						ids[root] = index;
						count--;
					}
				}
			}
			res.add(count);
		}
		return res;
	}

	public int findRoot(int[] ids, int i) {
		while (i != ids[i]) {
			ids[i] = ids[ids[i]];// find i's grandpa, reduce tree's height
			i = ids[i];
		}
		return i;
	}

	public void solve(char[][] board) {
		int m = board.length;
		if (m == 0)
			return;
		int n = board[0].length;
		// left
		for (int i = 0; i < m; i++) {
			if (board[i][0] == 'O') {
				dfsSolve(board, i, 0);
			}
		}
		// right
		for (int i = 0; i < m; i++) {
			if (board[i][n - 1] == 'O') {
				dfsSolve(board, i, n - 1);
			}
		}
		// top
		for (int i = 0; i < n; i++) {
			if (board[0][i] == 'O') {
				dfsSolve(board, 0, i);
			}
		}
		// bottom
		for (int i = 0; i < n; i++) {
			if (board[m - 1][i] == 'O') {
				dfsSolve(board, m - 1, i);
			}
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 'O')
					board[i][j] = 'X';
				if (board[i][j] == '#')
					board[i][j] = 'O';
			}
		}
	}

	public void dfsSolve(char[][] board, int i, int j) {
		if (i < 0 || i >= board.length || j < 0 || j >= board[0].length
				|| board[i][j] == 'X' || board[i][j] == '#')
			return;
		board[i][j] = '#';
		dfsSolve(board, i + 1, j);
		dfsSolve(board, i - 1, j);
		dfsSolve(board, i, j + 1);
		dfsSolve(board, i, j - 1);
	}

	public void bfsSolve(char[][] board, int i, int j, Queue<Integer> que) {
		int len = board.length;
		int wid = board[0].length;
		int loc = i * len + j;
		que.add(loc);
		while (!que.isEmpty()) {
			int cur = que.remove();
			int x = cur / len;
			int y = cur % len;
			board[x][y] = '#';
			if (x + 1 < len && board[x + 1][y] == 'O')
				que.add((x + 1) * len + y);
			if (x - 1 >= 0 && board[x - 1][y] == 'O')
				que.add((x - 1) * len + y);
			if (y + 1 < wid && board[x][y + 1] == 'O')
				que.add(x * len + y + 1);
			if (y - 1 >= 0 && board[x][y - 1] == 'O')
				que.add(x * len + y - 1);
		}
	}

	public void solve2(char[][] board) {
		int m = board.length;
		if (m == 0)
			return;
		int n = board[0].length;
		// left right
		for (int i = 0; i < m; i++) {
			bfsSolve(board, i, 0);
			bfsSolve(board, i, n - 1);
		}

		// top bottom
		for (int i = 0; i < n; i++) {
			bfsSolve(board, 0, i);
			bfsSolve(board, m - 1, i);
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 'O')
					board[i][j] = 'X';
				if (board[i][j] == '#')
					board[i][j] = 'O';
			}
		}
	}

	public void bfsSolve(char[][] board, int i, int j) {
		if (board[i][j] != 'O')
			return;
		board[i][j] = '#';
		Queue<Integer> que = new LinkedList<Integer>();
		int len = board.length;
		int wid = board[0].length;
		int loc = i * wid + j;
		que.add(loc);
		while (!que.isEmpty()) {
			int cur = que.remove();
			int x = cur / wid;
			int y = cur % wid;
			if (x + 1 < len && board[x + 1][y] == 'O') {
				que.add((x + 1) * len + y);
				board[x + 1][y] = '#';
			}
			if (x - 1 >= 0 && board[x - 1][y] == 'O') {
				que.add((x - 1) * len + y);
				board[x - 1][y] = '#';
			}
			if (y + 1 < wid && board[x][y + 1] == 'O') {
				que.add(x * len + y + 1);
				board[x][y + 1] = '#';
			}
			if (y - 1 >= 0 && board[x][y - 1] == 'O') {
				que.add(x * len + y - 1);
				board[x][y - 1] = '#';
			}
		}
	}

	public int mySqrt(int x) {
		if (x < 0)
			return Integer.MIN_VALUE;
		long i = 0, j = x;
		while (i <= j) {
			long mid = i + (j - i) / 2;
			if (mid * mid == x)
				return (int) mid;
			else if (mid * mid > x)
				j = mid - 1;
			else
				i = mid + 1;
		}
		return (int) j;
	}

	public int mySqrt2(int x) {
		if (x == 0)
			return 0;
		double lastY = 0;
		double y = 1;
		while (lastY != y) {
			lastY = y;
			y = (y + x / y) / 2;
		}
		return (int) y;
	}

	public double sqrt(double x) {
		if (x < 0)
			throw new IllegalArgumentException();
		double eps = 1.0e-10;
		double lo = 0.0;
		double hi = Math.max(1.0, x);
		double mid = (lo + hi) / 2.0;
		while (Math.abs(mid * mid - x) > eps) {
			if (mid * mid >= x)
				hi = mid;
			else
				lo = mid;
			mid = (lo + hi) / 2.0;
		}
		return mid;
	}

	public double sqrt2(double x) {
		if (x < 0)
			throw new IllegalArgumentException();
		double eps = 1.0e-10;
		if (x == 0.0)
			return 0.0;
		double pre = x / 2 + 1;
		double post = (pre + x / pre) / 2;
		while (Math.abs(post - pre) > eps) {
			pre = post;
			post = (pre + x / pre) / 2;
		}

		return pre;
	}

	public boolean isPerfectSquare(int num) {
		if (num < 1)
			return false;
		long res = num;
		while (res * res > num) {
			res = (res + num / res) / 2;
		}
		return res * res == num;
	}

	public double myPow(double x, int n) {
		if (n == 0)
			return 1;
		boolean neg = false;
		if (n < 0) {
			n = -n;
			neg = true;
		}
		double res = myPow(x, n / 2);
		if (n % 2 == 0)
			res *= res;
		else
			res *= res * x;

		return neg ? 1 / res : res;
	}

	public List<List<Integer>> combinationSum2Op(int[] candidates, int target) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		Arrays.sort(candidates);
		List<Integer> sol = new ArrayList<Integer>();
		combinationSum2(candidates, target, sol, res, 0);
		return res;
	}

	public void combinationSum2(int[] candidates, int target,
			List<Integer> sol, List<List<Integer>> res, int cur) {
		if (target == 0) {
			res.add(new ArrayList<Integer>(sol));
			return;
		}

		for (int i = cur; i < candidates.length; i++) {
			if (i > cur && candidates[i] == candidates[i - 1])
				continue;
			if (target == 0) {
				res.add(new ArrayList<Integer>(sol));
				return;
			}
			if (candidates[i] > target)
				break;
			sol.add(candidates[i]);
			combinationSum2(candidates, target - candidates[i], sol, res, i + 1);
			sol.remove(sol.size() - 1);
		}
	}

	public boolean isHappy(int n) {
		while (n >= 10) {
			int sum = 0;
			while (n > 0) {
				sum += Math.pow(n % 10, 2);
				n /= 10;
			}
			n = sum;
		}
		return n == 1 || n == 7;
	}

	public boolean isHappy2(int n) {
		HashSet<Integer> set = new HashSet<Integer>();
		while (n != 1) {
			if (set.contains(n))
				break;
			set.add(n);
			int sum = 0;
			while (n > 0) {
				sum += Math.pow(n % 10, 2);
				n /= 10;
			}
			n = sum;
		}
		return n == 1;
	}

	public int hIndex1_1(int[] citations) {
		if (citations.length == 0)
			return 0;
		Arrays.sort(citations);

		for (int i = 0; i < citations.length; i++) {
			if (citations[i] >= citations.length - i)
				return citations.length - i;
		}
		return citations[0] >= citations.length ? citations.length : 0;
	}

	public int hIndex1_2(int[] citations) {
		Arrays.sort(citations);
		int h = 0;
		for (int i = 0; i < citations.length; i++) {
			int curH = Math.min(citations[i], citations.length - i);
			h = Math.max(h, curH);
		}
		return h;
	}

	public int hIndex_ON(int[] citations) {
		int[] stats = new int[citations.length + 1];
		int n = citations.length;
		// 统计各个引用次数对应多少篇文章
		for (int i = 0; i < n; i++) {
			stats[citations[i] <= n ? citations[i] : n] += 1;
		}
		int sum = 0;
		// 找出最大的H指数
		for (int i = n; i > 0; i--) {
			// 引用大于等于i次的文章数量，等于引用大于等于i+1次的文章数量，加上引用等于i次的文章数量
			sum += stats[i];
			// 如果引用大于等于i次的文章数量，大于引用次数i，说明是H指数
			if (sum >= i) {
				return i;
			}
		}
		return 0;
	}

	public int hIndexII(int[] citations) {
		int n = citations.length;
		if (n == 0)
			return 0;
		int beg = 0;
		int end = citations.length - 1;

		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (citations[mid] < n - mid)
				beg = mid + 1;
			else
				end = mid - 1;
		}
		return n - beg;
	}

	public int numSquares(int n) {
		int[] dp = new int[n + 1];
		for (int i = 1; i <= n; i++) {
			dp[i] = i;
			for (int j = 1; j * j <= i; j++) {
				dp[i] = Math.min(dp[i - j * j] + 1, dp[i]);
			}
		}
		return dp[n];
	}

	private final String[] belowTen = new String[] { "", "One", "Two", "Three",
			"Four", "Five", "Six", "Seven", "Eight", "Nine" };
	private final String[] belowTwenty = new String[] { "Ten", "Eleven",
			"Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen",
			"Seventeen", "Eighteen", "Nineteen" };
	private final String[] belowHundred = new String[] { "", "Ten", "Twenty",
			"Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety" };

	public String numberToWords(int num) {
		if (num == 0)
			return "Zero";
		return convertNum(num);
	}

	public String convertNum(int num) {
		String result = "";
		if (num < 10)
			return belowTen[num];
		else if (num < 20)
			return belowTwenty[num - 10];
		else if (num < 100)
			result = belowHundred[num / 10] + " " + convertNum(num % 10);
		else if (num < 1000)
			result = convertNum(num / 100) + " Hundred "
					+ convertNum(num % 100);
		else if (num < 1000000)
			result = convertNum(num / 1000) + " Thousand "
					+ convertNum(num % 1000);
		else if (num < 1000000000)
			result = convertNum(num / 1000000) + " Million "
					+ convertNum(num % 1000000);
		else
			result = convertNum(num / 1000000000) + " Billion "
					+ convertNum(num % 1000000000);
		return result.trim();
	}

	private final String[] lessThan20 = { "", "One", "Two", "Three", "Four",
			"Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve",
			"Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen",
			"Eighteen", "Nineteen" };
	private final String[] tens = { "", "", "Twenty", "Thirty", "Forty",
			"Fifty", "Sixty", "Seventy", "Eighty", "Ninety" };
	private final String[] thousands = { "", "Thousand", "Million", "Billion" };

	public String numberToWords2(int num) {
		if (num == 0)
			return "Zero";
		String result = "";
		int i = 0;
		while (num > 0) {
			if (num % 1000 != 0) {
				result = helper(num % 1000) + thousands[i] + " " + result;
			}
			num /= 1000;
			i++;
		}
		return result.trim();
	}

	private String helper(int num) {
		if (num == 0) {
			return "";
		} else if (num < 20) {
			return lessThan20[num] + " ";
		} else if (num < 100) {
			return tens[num / 10] + " " + helper(num % 10);
		} else {
			return lessThan20[num / 100] + " Hundred " + helper(num % 100);
		}
	}

	public boolean exist(char[][] board, String word) {
		if (word.length() == 0)
			return true;
		int m = board.length;
		if (m == 0)
			return false;
		int n = board[0].length;
		boolean[][] used = new boolean[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == word.charAt(0)) {
					boolean res = dfsSearch(board, used, word, i, j, 0);
					if (res)
						return true;
				}
			}
		}
		return false;
	}

	public boolean dfsSearch(char[][] board, boolean[][] used, String word,
			int i, int j, int cur) {
		if (cur == word.length())
			return true;
		if (i < 0 || i >= board.length || j < 0 || j >= board[0].length
				|| used[i][j] || word.charAt(cur) != board[i][j])
			return false;
		used[i][j] = true;
		boolean res = dfsSearch(board, used, word, i + 1, j, cur + 1)
				|| dfsSearch(board, used, word, i - 1, j, cur + 1)
				|| dfsSearch(board, used, word, i, j + 1, cur + 1)
				|| dfsSearch(board, used, word, i, j - 1, cur + 1);
		used[i][j] = false;
		return res;
	}

	public List<String> findWords(char[][] board, String[] words) {
		List<String> res = new ArrayList<String>();
		int m = board.length;
		if (m == 0)
			return res;
		int n = board[0].length;
		boolean[][] used = new boolean[m][n];
		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			for (int j = 0; j < m; j++) {
				for (int k = 0; k < n; k++) {
					if (board[j][k] == word.charAt(0)) {
						if (dfsBoard(board, word, j, k, used, 0)) {
							if (!res.contains(word))
								res.add(word);
						}
					}
				}
			}
		}
		return res;
	}

	public boolean dfsBoard(char[][] board, String word, int i, int j,
			boolean[][] used, int cur) {
		if (cur == word.length())
			return true;
		if (i < 0 || i >= board.length || j < 0 || j >= board[0].length
				|| used[i][j] || board[i][j] != word.charAt(cur))
			return false;
		used[i][j] = true;
		boolean res = dfsBoard(board, word, i + 1, j, used, cur + 1)
				|| dfsBoard(board, word, i - 1, j, used, cur + 1)
				|| dfsBoard(board, word, i, j + 1, used, cur + 1)
				|| dfsBoard(board, word, i, j - 1, used, cur + 1);
		used[i][j] = false;
		return res;
	}

	public List<String> findWords2(char[][] board, String[] words) {
		Set<String> res = new HashSet<String>();
		int m = board.length;
		if (m == 0)
			return new ArrayList<String>();
		int n = board[0].length;
		Trie trie = new Trie();
		for (String word : words) {
			trie.insert(word);
		}
		boolean[][] used = new boolean[m][n];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				dfsBoard(board, trie, i, j, used, "", res);
			}
		}
		return new ArrayList<String>(res);
	}

	public void dfsBoard(char[][] board, Trie trie, int i, int j,
			boolean[][] used, String word, Set<String> res) {
		if (i < 0 || i >= board.length || j < 0 || j >= board[0].length
				|| used[i][j])
			return;
		word += board[i][j];
		if (!trie.startsWith(word))
			return;
		if (trie.search(word))
			res.add(word);
		used[i][j] = true;
		dfsBoard(board, trie, i + 1, j, used, word, res);
		dfsBoard(board, trie, i - 1, j, used, word, res);
		dfsBoard(board, trie, i, j + 1, used, word, res);
		dfsBoard(board, trie, i, j - 1, used, word, res);
		used[i][j] = false;
	}

	public String simplifyPath(String path) {
		String[] strs = path.split("/");
		Stack<String> stk = new Stack<String>();
		for (String s : strs) {
			if (s.equals(".") || s.isEmpty())
				continue;
			else if (s.equals("..")) {
				if (!stk.isEmpty())
					stk.pop();
			} else
				stk.push(s);
		}
		String res = "";
		if (stk.isEmpty())
			return "/";
		while (!stk.isEmpty()) {
			res = "/" + stk.pop() + res;
		}
		return res;
	}

	public int shortestWordDistance(String[] words, String word1, String word2) {
		int idx1 = -1, idx2 = -1, minDis = Integer.MAX_VALUE;
		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			if (word1.equals(word2)) {
				if (word.equals(word1)) {
					if (idx1 > idx2)
						idx2 = i;
					else
						idx1 = i;
				}
			} else {
				if (word.equals(word1))
					idx1 = i;
				else if (word.equals(word2))
					idx2 = i;
			}
			if (idx1 != -1 && idx2 != -1)
				minDis = Math.min(minDis, Math.abs(idx1 - idx2));
		}
		return minDis;
	}

	// linkedin interview
	public int deepSum(List<Object> list) {
		if (list.size() == 0)
			return 0;
		return deepSum(list, 1);
	}

	public int deepSum(List<Object> list, int level) {
		int sum = 0;
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i) instanceof List)
				sum += deepSum((List<Object>) list.get(i), level + 1);
			else
				sum += (int) list.get(i) * level;
		}
		return sum;
	}

	public int reverseDeepSum(List<Object> list) {
		int depth = getDepth(list);
		return reverseDeepSum(list, depth);
	}

	public int reverseDeepSum(List<Object> list, int depth) {
		if (list.size() == 0)
			return 0;
		int sum = 0;
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i) instanceof Integer)
				sum += (Integer) list.get(i) * depth;
			else
				sum += reverseDeepSum((List) list.get(i), depth - 1);
		}
		return sum;
	}

	@SuppressWarnings("unchecked")
	public int getDepth(List<Object> list) {
		if (list.size() == 0)
			return 0;
		int depth = 1;
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i) instanceof List)
				depth = Math.max(depth,
						1 + getDepth((List<Object>) list.get(i)));
		}
		return depth;
	}

	public TreeNode UpsideDownBinaryTree(TreeNode root) {
		if (root == null || root.left == null)
			return root;
		TreeNode res = UpsideDownBinaryTree(root.left);
		root.left.left = root.right;
		root.left.right = root;
		root.left = null;
		root.right = null;
		return res;
	}

	public TreeNode UpsideDownBinaryTree2(TreeNode root) {
		if (root == null)
			return null;
		TreeNode node = root, parent = null, right = null;
		while (node != null) {
			TreeNode left = node.left;
			node.left = right;
			right = node.right;
			node.right = parent;
			parent = node;
			node = left;
		}
		return parent;
	}

	public void inorder(TreeNode root) {
		if (root == null)
			return;
		inorder(root.left);
		System.out.print(root.val + " ");
		inorder(root.right);
	}

	public int maxPoints(Point[] points) {
		if (points.length < 3)
			return points.length;
		int max = 1;
		for (int i = 0; i < points.length; i++) {
			Point p1 = points[i];
			HashMap<Double, Integer> map = new HashMap<Double, Integer>();
			int dup = 1;
			int vertical = 0;
			for (int j = i + 1; j < points.length; j++) {
				Point p2 = points[j];
				if (p1.x == p2.x) {
					if (p1.y == p2.y)
						dup++;
					else
						vertical++;
				} else {
					double k = p1.y == p2.y ? 0.0 : 1.0 * (p1.y - p2.y)
							/ (p1.x - p2.x);
					if (map.containsKey(k)) {
						map.put(k, map.get(k) + 1);
					} else {
						map.put(k, 1);
					}
				}
			}
			for (double k : map.keySet()) {
				max = Math.max(max, map.get(k) + dup);
			}
			max = Math.max(max, dup + vertical);
		}
		return max;
	}

	public int maxPoints2(Point[] points) {
		if (points.length < 3)
			return points.length;
		int max = 2;
		for (int i = 0; i < points.length - 1; i++) {
			Map<Double, Integer> map = new HashMap<Double, Integer>();
			int dup = 0, vertical = 1;
			Point p1 = points[i];
			int curmax = 1;
			for (int j = i + 1; j < points.length; j++) {
				Point p2 = points[j];
				if (p1.x == p2.x) {
					if (p1.y == p2.y)
						dup++;
					else
						vertical++;
				} else {
					double k = p1.y == p2.y ? 0.0 : 1.0 * (p1.y - p2.y)
							/ (p1.x - p2.x);
					if (map.containsKey(k))
						map.put(k, map.get(k) + 1);
					else
						map.put(k, 2);
					if (map.get(k) > curmax)
						curmax = map.get(k);
				}
			}
			max = Math.max(max, Math.max(curmax + dup, vertical + dup));
		}
		return max;
	}

	public int maxPoints3(Point[] points) {
		if (points.length < 3)
			return points.length;
		int res = 1;
		for (int i = 0; i < points.length - 1; i++) {
			int dup = 1;
			Map<Double, Integer> map = new HashMap<Double, Integer>();
			for (int j = i + 1; j < points.length; j++) {
				if (points[i].x == points[j].x && points[i].y == points[j].y)
					dup++;
				else if (points[i].x == points[j].x) {
					if (!map.containsKey(Double.MAX_VALUE))
						map.put(Double.MAX_VALUE, 0);
					map.put(Double.MAX_VALUE, map.get(Double.MAX_VALUE) + 1);
				} else {
					double k = points[i].y == points[j].y ? 0.0
							: (points[i].y - points[j].y)
									/ (double) (points[i].x - points[j].x);
					if (!map.containsKey(k))
						map.put(k, 0);
					map.put(k, map.get(k) + 1);
				}
			}
			int localMax = 0;
			for (int v : map.values())
				localMax = Math.max(localMax, v);
			localMax += dup;
			res = Math.max(res, localMax);
		}
		return res;
	}

	public int minDistance(String word1, String word2) {
		int n1 = word1.length();
		int n2 = word2.length();
		int[][] dp = new int[n1 + 1][n2 + 1];
		for (int i = 0; i <= n1; i++) {
			dp[i][0] = i;
		}
		for (int i = 0; i <= n2; i++) {
			dp[0][i] = i;
		}

		for (int i = 1; i <= n1; i++) {
			for (int j = 1; j <= n2; j++) {
				if (word1.charAt(i - 1) == word2.charAt(j - 1))
					dp[i][j] = dp[i - 1][j - 1];
				else
					dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]),
							dp[i - 1][j - 1]) + 1;
			}
		}
		return dp[n1][n2];
	}

	public boolean isOneEditDistance(String s, String t) {
		for (int i = 0; i < Math.min(s.length(), t.length()); i++) {
			if (s.charAt(i) != t.charAt(i)) {
				if (s.length() == t.length())
					return s.substring(i + 1).equals(t.substring(i + 1));
				else if (s.length() < t.length())
					return s.substring(i).equals(t.substring(i + 1));
				else
					return s.substring(i + 1).equals(i);
			}
		}
		return Math.abs(s.length() - t.length()) == 1;
	}

	public boolean isOneEditDistance2(String s, String t) {
		if (Math.abs(s.length() - t.length()) > 1)
			return false;
		int i = 0, j = 0, error = 0;
		while (i < s.length() && j < t.length()) {
			if (s.charAt(i) != t.charAt(j)) {
				error++;
				if (s.length() > t.length())
					j--;
				else if (s.length() < t.length())
					i--;
			}
			i++;
			j++;
		}
		return error == 1 || (error == 0 && s.length() != t.length());
	}

	public boolean isOneEditDistance3(String s, String t) {
		if (Math.abs(s.length() - t.length()) > 1)
			return false;
		boolean miss = false;
		int i = 0, j = 0;
		while (i < s.length() && j < t.length()) {
			if (s.charAt(i) != t.charAt(i)) {
				if (miss)
					return false;
				miss = true;
				if (s.length() < t.length())
					i--;
			}
			i++;
			j++;
		}
		return miss || s.length() != t.length();
	}

	public String rearrange(String s, int d) {
		if (s.length() < 2)
			return s;

		int count = 0;
		CharFreq[] cf = new CharFreq[256];
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (cf[c] == null) {
				cf[c] = new CharFreq(c);
				count++;
			} else
				cf[c].freq++;
		}

		PriorityQueue<CharFreq> heap = new PriorityQueue<CharFreq>(count,
				new Comparator<CharFreq>() {

					@Override
					public int compare(CharFreq o1, CharFreq o2) {
						// TODO Auto-generated method stub
						return o2.freq - o1.freq;
					}

				});

		for (int i = 0; i < cf.length; i++) {
			if (cf[i] != null) {
				heap.offer(cf[i]);
			}
		}
		char[] res = s.toCharArray();
		for (int i = 0; i < res.length; i++) {
			res[i] = '#';
		}
		for (int i = 0; i < count; i++) {
			int p = i;
			while (p < res.length && res[p] != '#')
				p++;
			CharFreq charFreq = heap.poll();
			for (int k = 0; k < charFreq.freq; k++) {
				if (p + k * d >= s.length()) {
					// throw expception
					System.out.println("cannot rearrange!");
					break;
				}
				res[p + k * d] = charFreq.c;
			}
		}
		return new String(res);
	}

	public int secondMin(TreeNode root) {
		if (root == null)
			return Integer.MAX_VALUE;
		if (root.left != null && root.right != null) {
			if (root.left.val == root.val) {
				return Math.min(secondMin(root.left), root.right.val);
			} else {
				return Math.min(secondMin(root.right), root.left.val);
			}
		}
		return Integer.MAX_VALUE;
	}

	public int findMinAvgSubarray(int[] nums, int k) {
		if (nums.length < k)
			return Integer.MAX_VALUE;
		int cursum = 0;
		for (int i = 0; i < k; i++)
			cursum += nums[i];
		int minSum = cursum;
		int end = k - 1;
		for (int i = k; i < nums.length; i++) {
			cursum += nums[i] - nums[i - k];
			if (cursum < minSum) {
				minSum = cursum;
				end = i;
			}
		}
		System.out.println("subarry from " + (end - k + 1) + " to " + end);
		return minSum / k;
	}

	public boolean canPermutePalindrome(String s) {
		Set<Character> set = new HashSet<Character>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (set.contains(c))
				set.remove(c);
			else
				set.add(c);
		}
		return set.size() < 2;
	}

	public boolean canPermutePalindrome2(String s) {
		int[] counts = new int[256];
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (counts[c] > 0)
				counts[c]--;
			else
				counts[c]++;
		}
		int count = 0;
		for (int i = 0; i < 256; i++) {
			if (counts[i] != 0)
				count++;
		}
		return count <= 1;
	}

	public List<String> generatePalindromes(String s) {
		List<String> res = new ArrayList<String>();
		Map<Character, Integer> map = new HashMap<Character, Integer>();
		Set<Character> set = new HashSet<Character>();
		// int odd=0;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			map.put(c, map.containsKey(c) ? map.get(c) + 1 : 1);
			// odd += map.get(c) % 2 != 0 ? 1 : -1;
			if (set.contains(c))
				set.remove(c);
			else
				set.add(c);
		}
		if (set.size() > 1)
			return res;
		List<Character> lst = new ArrayList<Character>();
		String mid = "";
		for (char c : map.keySet()) {
			int count = map.get(c);
			if (count % 2 != 0)
				mid += c;
			for (int i = 0; i < count / 2; i++) {
				lst.add(c);
			}
		}
		boolean[] used = new boolean[lst.size()];
		getPermutations(mid, lst, used, new StringBuilder(), res);
		return res;
	}

	public void getPermutations(String mid, List<Character> lst,
			boolean[] used, StringBuilder sb, List<String> res) {
		if (sb.length() == lst.size()) {
			res.add(sb.toString() + mid + sb.reverse().toString());
			sb.reverse();
			return;
		}

		for (int i = 0; i < lst.size(); i++) {
			if (i != 0 && lst.get(i) == lst.get(i - 1) && !used[i - 1])
				continue;
			if (!used[i]) {
				used[i] = true;
				sb.append(lst.get(i));
				getPermutations(mid, lst, used, sb, res);
				used[i] = false;
				sb.deleteCharAt(sb.length() - 1);
			}
		}
	}

	public String countAndSay(int n) {
		if (n < 1)
			return "";
		String num = "1";
		for (int i = 1; i < n; i++) {
			char c = num.charAt(0);
			int count = 1;
			String tmp = "";
			for (int j = 1; j < num.length(); j++) {
				if (num.charAt(j) == c)
					count++;
				else {
					tmp += "" + count + c;
					c = num.charAt(j);
					count = 1;
				}
				System.out.println("sss " + num);
			}
			tmp += "" + count + c;
			num = tmp;
		}
		return num;
	}

	public int minPathSum(int[][] grid) {
		int m = grid.length;
		if (m == 0)
			return 0;
		int n = grid[0].length;
		int[][] dp = new int[m][n];
		dp[0][0] = grid[0][0];
		for (int i = 1; i < m; i++) {
			dp[i][0] = dp[i - 1][0] + grid[i][0];
		}
		for (int j = 1; j < n; j++) {
			dp[0][j] = dp[0][j - 1] + grid[0][j];
		}

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
			}
		}
		return dp[m - 1][n - 1];
	}

	// O(1) space
	public int minPathSum2(int[][] grid) {
		int row = grid.length;
		if (row == 0) {
			return 0;
		}
		int col = grid[0].length;

		for (int i = 1; i < row; i++) {
			grid[i][0] += grid[i - 1][0];
		}

		for (int j = 1; j < col; j++) {
			grid[0][j] += grid[0][j - 1];
		}

		for (int i = 1; i < row; i++) {
			for (int j = 1; j < col; j++) {
				grid[i][j] = grid[i][j]
						+ Math.min(grid[i - 1][j], grid[i][j - 1]);
			}
		}
		return grid[row - 1][col - 1];
	}

	public int uniquePaths(int m, int n) {
		int[][] dp = new int[m][n];
		for (int i = 0; i < m; i++) {
			dp[i][0] = 1;
		}
		for (int i = 0; i < n; i++) {
			dp[0][i] = 1;
		}

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
			}
		}
		return dp[m - 1][n - 1];
	}

	public int lengthOfLastWord(String s) {
		s = s.trim();
		int count = 0;
		int i = s.length() - 1;
		while (i >= 0 && s.charAt(i--) != ' ') {
			count++;
		}
		return count;
	}

	public int numDecodings1(String s) {
		if (s.length() == 0)
			return 0;
		int[] count = { 0 };
		numDecodings(s, count);
		return count[0];
	}

	public void numDecodings(String s, int[] count) {
		if (s.length() == 0)
			count[0]++;
		for (int i = 0; i <= 1 && i < s.length(); i++) {
			if (isValidNum(s.substring(0, i + 1)))
				numDecodings(s.substring(i + 1), count);
		}
	}

	public int numDecodings2(String s) {

		int n = s.length();
		if (n == 0)
			return 0;
		int[] dp = new int[n + 1];
		dp[0] = 1;
		dp[1] = isValidNum(s.substring(0, 1)) ? 1 : 0;

		for (int i = 2; i <= n; i++) {
			if (isValidNum(s.substring(i - 1, i)))
				dp[i] += dp[i - 1];
			if (isValidNum(s.substring(i - 2, i)))
				dp[i] += dp[i - 2];
		}
		return dp[n];
	}

	public boolean isValidNum(String s) {
		if (s.charAt(0) == '0')
			return false;
		int num = Integer.parseInt(s);
		return num >= 1 && num <= 26;
	}

	public void rotate(int[][] matrix) {
		int m = matrix.length;
		if (m == 0)
			return;
		int n = matrix[0].length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < i; j++) {
				int t = matrix[i][j];
				matrix[i][j] = matrix[j][i];
				matrix[j][i] = t;
			}
		}

		for (int i = 0; i < m; i++) {
			int beg = 0;
			int end = n - 1;
			while (beg < end) {
				int t = matrix[i][beg];
				matrix[i][beg++] = matrix[i][end];
				matrix[i][end--] = t;
			}
		}
	}

	// find celebrity
	public boolean knows(int a, int b) {
		return true;
	}

	public int findCelebrity(int n) {
		int candidate = 0;
		for (int i = 1; i < n; i++) {
			if (knows(candidate, i))
				candidate = i;
		}
		for (int i = 0; i < n; i++) {
			if (i != candidate && (knows(candidate, i) || !knows(i, candidate)))
				return -1;
		}
		return candidate;
	}

	public int findCelebrity2(int n) {
		// base case
		if (n <= 0)
			return -1;
		if (n == 1)
			return 0;

		Stack<Integer> stack = new Stack<>();

		// put all people to the stack
		for (int i = 0; i < n; i++)
			stack.push(i);

		int a = 0, b = 0;

		while (stack.size() > 1) {
			a = stack.pop();
			b = stack.pop();

			if (knows(a, b))
				// a knows b, so a is not the celebrity, but b may be
				stack.push(b);
			else
				// a doesn't know b, so b is not the celebrity, but a may be
				stack.push(a);
		}

		// double check the potential celebrity
		int c = stack.pop();

		for (int i = 0; i < n; i++)
			// c should not know anyone else
			if (i != c && (knows(c, i) || !knows(i, c)))
				return -1;

		return c;
	}

	public int longestConsecutive(int[] nums) {
		if (nums.length < 2)
			return nums.length;
		HashSet<Integer> set = new HashSet<Integer>();
		int max = 1;
		for (int i = 0; i < nums.length; i++) {
			set.add(nums[i]);
		}

		for (int i = 0; i < nums.length; i++) {
			int num = nums[i];
			int count = 0;
			while (set.contains(num)) {
				count++;
				set.remove(num--);
			}
			num = nums[i] + 1;
			while (set.contains(num)) {
				count++;
				set.remove(num++);
			}
			max = Math.max(max, count);
		}
		return max;
	}

	public int longestConsecutive2(int[] nums) {
		if (nums.length < 2)
			return nums.length;
		HashSet<Integer> set = new HashSet<Integer>();
		int max = 1;
		for (int i = 0; i < nums.length; i++) {
			set.add(nums[i]);
		}

		for (int i = 0; i < nums.length; i++) {
			int num = nums[i];
			max = Math.max(max,
					countDesc(set, num, true) + countDesc(set, num + 1, false));
		}
		return max;
	}

	public int countDesc(Set<Integer> set, int num, boolean desc) {
		int count = 0;
		while (set.contains(num)) {
			count++;
			set.remove(num);
			if (desc)
				num--;
			else
				num++;
		}
		return count;
	}

	public int longesetConsecutive(TreeNode root) {
		if (root == null)
			return 0;
		int[] max = { 0 };
		// helper(root, root.left, 1, max);
		// helper(root, root.right, 1, max);
		helper(null, root, 0, max);
		return max[0];
	}

	public void helper(TreeNode parent, TreeNode cur, int curLen, int[] max) {
		if (cur == null)
			return;
		curLen = parent == null || cur.val != parent.val + 1 ? 1 : curLen + 1;
		max[0] = Math.max(max[0], curLen);
		helper(cur, cur.left, curLen, max);
		helper(cur, cur.right, curLen, max);
	}

	public int longestConsecutive(TreeNode root) {
		if (root == null)
			return 0;
		return Math.max(longestConsecutive(root.left, 1, root.val),
				longestConsecutive(root.right, 1, root.val));
	}

	// google interview
	// longest consecutive sequence in a tree
	// followup: not binary tree

	public int longestConsecutiveFollowup(Node root) {
		if (root == null)
			return 0;
		int[] max = { 1 };
		longestConsecutiveFollowupUtil(root, 1, max);
		return max[0];
	}

	public void longestConsecutiveFollowupUtil(Node root, int cur, int[] max) {
		max[0] = Math.max(cur, max[0]);
		if (root == null) {
			return;
		}
		if (root.children != null) {
			for (Node child : root.children) {
				if (child != null) {
					if (child.val == root.val + 1)
						longestConsecutiveFollowupUtil(child, cur + 1, max);
					else
						longestConsecutiveFollowupUtil(child, 1, max);
				}
			}
		}
	}

	public int longestConsecutive(TreeNode root, int count, int val) {
		if (root == null)
			return count;
		count = (root.val - val == 1) ? count + 1 : 1;
		int left = longestConsecutive(root.left, count, root.val);
		int right = longestConsecutive(root.right, count, root.val);

		return Math.max(Math.max(left, right), count);
	}

	public int deepestNode(TreeNode root) {
		if (root == null)
			return Integer.MAX_VALUE;
		int[] deepest = { 0 };
		int[] val = { root.val };
		deepestNode(root, 0, val, deepest);
		return val[0];
	}

	public void deepestNode(TreeNode root, int dep, int[] val, int[] deepest) {
		if (root == null)
			return;
		if (dep > deepest[0]) {
			val[0] = root.val;
			deepest[0] = dep;
		}

		deepestNode(root.left, dep + 1, val, deepest);
		deepestNode(root.right, dep + 1, val, deepest);
	}

	public boolean isValidSudoku(char[][] board) {
		for (int i = 0; i < 9; i++) {
			boolean[] rows = new boolean[10];
			boolean[] cols = new boolean[10];
			for (int j = 0; j < 9; j++) {
				int num = board[i][j] - '0';
				if (num > 0 && num <= 9) {
					if (rows[num])
						return false;
					rows[num] = true;
				}
				num = board[j][i] - '0';
				if (num > 0 && num <= 9) {
					if (cols[num])
						return false;
					cols[num] = true;
				}
			}
		}
		// method 2
		// rule3, sub-box
		// for(int i=0; i<3; i++){
		// for(int j=0; j<3; j++){// for each sub-box
		// HashSet<Character> test = new HashSet<Character>();
		// for(int m=i*3; m<i*3+3; m++){//row
		// for(int n=j*3; n<j*3+3; n++){//column
		// if(board[m][n]!='.' && !test.add(board[m][n])) return false;
		// }
		// }
		// }
		// }

		// method 2
		// Check for each sub-grid
		// for (int k = 0; k < 9; k++) {
		// for (int i = k/3*3; i < k/3*3+3; i++) {
		// for (int j = (k%3)*3; j < (k%3)*3+3; j++) {
		// if (board[i][j] == '.')
		// continue;
		// if (set.contains(board[i][j]))
		// return false;
		// set.add(board[i][j]);
		// }
		// }
		// set.clear();
		// }

		for (int i = 0; i < 9; i += 3) {
			for (int j = 0; j < 9; j += 3) {
				boolean[] grid = new boolean[10];
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3; l++) {
						int num = board[i + k][j + l] - '0';
						if (num > 0 && num <= 9) {
							if (grid[num])
								return false;
							grid[num] = true;
						}
					}
				}
			}
		}
		return true;
	}

	public List<List<String>> solveNQueens(int n) {
		List<List<String>> res = new ArrayList<List<String>>();
		int[] rows = new int[n];
		solveNQueens(0, n, rows, res);
		return res;
	}

	public void solveNQueens(int cur, int n, int[] rows, List<List<String>> res) {
		if (cur == n) {
			printBoard(rows, res);
			return;
		}
		for (int i = 0; i < n; i++) {
			rows[cur] = i;
			if (isValid(rows, cur)) {
				solveNQueens(cur + 1, n, rows, res);
			}
		}
	}

	// 假设有两个皇后被放置在（i，j）和（k，l）的位置上，明显，当且仅当|i-k|=|j-l| 时，两个皇后才在同一条对角线上。

	public boolean isValid(int[] rows, int cur) {
		for (int i = 0; i < cur; i++) {
			if (rows[i] == rows[cur]
					|| Math.abs(rows[i] - rows[cur]) == cur - i)
				return false;
		}
		return true;
	}

	public void printBoard(int[] rows, List<List<String>> res) {
		List<String> sol = new ArrayList<String>();
		int n = rows.length;
		for (int i = 0; i < n; i++) {
			String row = "";
			for (int j = 0; j < n; j++) {
				if (rows[i] == j)
					row += "Q";
				else
					row += ".";
			}
			sol.add(row);
		}
		res.add(sol);
	}

	public int totalNQueens(int n) {
		int[] res = { 0 };
		int[] cols = new int[n];
		dfsTotalNQueens(0, n, cols, res);
		return res[0];
	}

	public void dfsTotalNQueens(int cur, int n, int[] cols, int[] res) {
		if (cur == n) {
			res[0]++;
			return;
		}

		for (int i = 0; i < n; i++) {
			cols[cur] = i;
			if (isValid(cur, cols))
				dfsTotalNQueens(cur + 1, n, cols, res);
		}
	}

	public boolean isValid(int cur, int[] cols) {
		for (int i = 0; i < cur; i++) {
			if (cols[i] == cols[cur]
					|| Math.abs(cols[i] - cols[cur]) == cur - i)
				return false;
		}
		return true;
	}

	public void solveSudoku(char[][] board) {
		List<int[]> empty = new ArrayList<int[]>();
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[0].length; j++) {
				if (board[i][j] == '.') {
					int[] dot = { i, j };
					empty.add(dot);
				}
			}
		}

		dfsSolveSudoku(0, board, empty);
	}

	public boolean dfsSolveSudoku(int cur, char[][] board, List<int[]> empty) {
		if (cur == empty.size()) {
			return true;
		}
		int row = empty.get(cur)[0];
		int col = empty.get(cur)[1];
		for (int i = 1; i <= 9; i++) {
			if (isValidSudoku(i, row, col, board)) {
				board[row][col] = (char) ('0' + i);
				if (dfsSolveSudoku(cur + 1, board, empty))
					return true;
				board[row][col] = '.';
			}
		}
		return false;
	}

	public boolean isValidSudoku(int val, int row, int col, char[][] board) {
		for (int i = 0; i < 9; i++) {
			if (board[row][i] == '0' + val)
				return false;
			if (board[i][col] == '0' + val)
				return false;
			int c_row = row / 3 * 3 + i / 3;
			int c_col = col / 3 * 3 + i % 3;
			if (board[c_row][c_col] == '0' + val)
				return false;
		}
		return true;
	}

	public List<Integer> grayCode(int n) {
		List<Integer> res = new ArrayList<Integer>();
		if (n == 0) {
			res.add(0);
			return res;
		}

		List<Integer> partial = grayCode(n - 1);
		res.addAll(partial);
		for (int i = partial.size() - 1; i >= 0; i--) {
			res.add(partial.get(i) + (1 << (n - 1)));
		}
		return res;
	}

	public boolean canWinNim(int n) {
		return (n % 4) != 0;
	}

	public List<List<Integer>> generate(int numRows) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (numRows <= 0)
			return res;
		int[][] dp = new int[numRows][numRows];
		for (int i = 0; i < numRows; i++) {
			dp[i][0] = 1;
		}

		for (int i = 1; i < numRows; i++) {
			for (int j = 1; j <= i; j++) {
				if (j == i)
					dp[i][j] = 1;
				else
					dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1];
			}
		}

		for (int i = 0; i < numRows; i++) {
			List<Integer> row = new ArrayList<Integer>();
			for (int j = 0; j <= i; j++) {
				row.add(dp[i][j]);
			}
			res.add(row);
		}
		return res;
	}

	public List<List<Integer>> generate2(int numRows) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (numRows <= 0)
			return res;
		List<Integer> pre = new ArrayList<Integer>();
		pre.add(1);
		res.add(pre);
		for (int i = 2; i <= numRows; i++) {
			List<Integer> cur = new ArrayList<Integer>();
			cur.add(1);// first element
			for (int j = 0; j < pre.size() - 1; j++) {
				cur.add(pre.get(j) + pre.get(j + 1));// middle elements
			}
			cur.add(1);// last element
			res.add(cur);
			pre = cur;
		}
		return res;
	}

	public List<List<Integer>> generate3(int numRows) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> row, pre = null;
		for (int i = 0; i < numRows; i++) {
			row = new ArrayList<Integer>();
			for (int j = 0; j <= i; j++) {
				if (j == 0 || j == i)
					row.add(1);
				else
					row.add(pre.get(j) + pre.get(j - 1));
			}
			res.add(row);
			pre = row;
		}
		return res;
	}

	public List<Integer> getRow(int rowIndex) {
		List<Integer> res = new ArrayList<Integer>();
		if (rowIndex < 0)
			return res;
		res.add(1);

		for (int i = 2; i <= rowIndex + 1; i++) {
			List<Integer> cur = new ArrayList<Integer>();
			cur.add(1);
			for (int j = 0; j < res.size() - 1; j++) {
				cur.add(res.get(j) + res.get(j + 1));
			}
			cur.add(1);
			res = cur;
		}
		return res;
	}

	public List<Integer> getRow2(int rowIndex) {
		List<Integer> res = new ArrayList<Integer>();
		if (rowIndex < 0)
			return res;
		res.add(1);

		for (int i = 1; i <= rowIndex; i++) {
			for (int j = res.size() - 2; j >= 0; j--) {
				res.set(j + 1, res.get(j) + res.get(j + 1));
			}
			res.add(1);
		}
		return res;
	}

	/*
	 * 实际上，我们可以根据抽屉原理简化刚才的暴力法。我们不一定要依次选择数，然后看是否有这个数的重复数，我们可以用二分法先选取n/2，
	 * 按照抽屉原理，整个数组中如果小于等于n/2的数的数量大于n/2，说明1到n/2这个区间是肯定有重复数字的。比如6个抽屉，
	 * 如果有7个袜子要放到抽屉里，那肯定有一个抽屉至少两个袜子。这里抽屉就是1到n/2的每一个数，而袜子就是整个数组中小于等于n/2的那些数。
	 * 这样我们就能知道下次选择的数的范围
	 * ，如果1到n/2区间内肯定有重复数字，则下次在1到n/2范围内找，否则在n/2到n范围内找。下次找的时候，还是找一半。 注意
	 * 
	 * 我们比较的mid而不是nums[mid] 因为mid是下标，所以判断式应为cnt > mid，最后返回min
	 */
	public int findDuplicate(int[] nums) {
		int beg = 0, end = nums.length - 1;
		while (beg <= end) {
			int mid = beg + (end - beg) / 2;
			int count = 0;
			for (int i = 0; i < nums.length; i++) {
				if (nums[i] <= mid)
					count++;
			}
			if (count <= mid)
				beg = mid + 1;
			else
				end = mid - 1;
		}
		return beg;
	}

	/*
	 * 映射找环法
	 * 
	 * 复杂度
	 * 
	 * 时间 O(N) 空间 O(1)
	 * 
	 * 思路
	 * 
	 * 假设数组中没有重复，那我们可以做到这么一点，就是将数组的下标和1到n每一个数一对一的映射起来。比如数组是213,则映射关系为0->2, 1->1,
	 * 2->3。 假设这个一对一映射关系是一个函数f(n)，其中n是下标，f(n)是映射到的数。如果我们从下标为0出发，
	 * 根据这个函数计算出一个值，以这个值为新的下标，再用这个函数计算，以此类推，直到下标超界。
	 * 实际上可以产生一个类似链表一样的序列。比如在这个例子中有两个下标的序列，0->2->3。
	 * 
	 * 但如果有重复的话，这中间就会产生多对一的映射，比如数组2131,则映射关系为0->2, {1，3}->1, 2->3。
	 * 这样，我们推演的序列就一定会有环路了，这里下标的序列是0->2->3->1->1->1->1->...，而环的起点就是重复的数。
	 * 
	 * 所以该题实际上就是找环路起点的题，和Linked List Cycle II一样。
	 * 我们先用快慢两个下标都从0开始，快下标每轮映射两次，慢下标每轮映射一次，直到两个下标再次相同。
	 * 这时候保持慢下标位置不变，再用一个新的下标从0开始，这两个下标都继续每轮映射一次，当这两个下标相遇时，就是环的起点，也就是重复的数。
	 * 对这个找环起点算法不懂的，请参考Floyd's Algorithm。
	 */
	public int findDuplicate2(int[] nums) {
		int slow = 0, fast = 0;
		do {
			slow = nums[slow];
			fast = nums[nums[fast]];
		} while (slow != fast);

		fast = 0;
		while (fast != slow) {
			fast = nums[fast];
			slow = nums[slow];
		}
		return slow;
	}

	public int minimumTotal(List<List<Integer>> triangle) {
		int n = triangle.size();
		if (n == 0)
			return 0;
		int m = triangle.get(n - 1).size();
		int[][] dp = new int[n][m];
		dp[0][0] = triangle.get(0).get(0);
		for (int i = 1; i < n; i++) {
			dp[i][0] = dp[i - 1][0] + triangle.get(i).get(0);
		}

		for (int i = 1; i < n; i++) {
			for (int j = 1; j <= i; j++) {
				if (j == i)
					dp[i][j] = dp[i - 1][j - 1] + triangle.get(i).get(j);
				else
					dp[i][j] = Math.min(dp[i - 1][j], dp[i - 1][j - 1])
							+ triangle.get(i).get(j);
			}
		}

		int min = Integer.MAX_VALUE;
		for (int i = 0; i < m; i++) {
			min = Math.min(dp[n - 1][i], min);
		}
		return min;
	}

	// O(n) space, top-down
	public int minimumTotal2(List<List<Integer>> triangle) {
		int n = triangle.size();
		if (n == 0)
			return 0;
		int m = triangle.get(n - 1).size();
		int[] dp = new int[m];
		dp[0] = triangle.get(0).get(0);

		for (int i = 1; i < n; i++) {
			for (int j = i; j >= 0; j--) {
				if (j == 0)
					dp[j] = dp[j] + triangle.get(i).get(0);
				else if (j == i)
					dp[j] = dp[j - 1] + triangle.get(i).get(j);
				else
					dp[j] = Math.min(dp[j - 1], dp[j]) + triangle.get(i).get(j);
			}
		}
		int min = Integer.MAX_VALUE;
		for (int i = 0; i < m; i++) {
			min = Math.min(min, dp[i]);
		}
		return min;
	}

	// O(n) space, bottom up
	public int minimumTotal3(List<List<Integer>> triangle) {
		int n = triangle.size();
		if (n == 0)
			return 0;
		int m = triangle.get(n - 1).size();
		int[] dp = new int[m];
		for (int i = 0; i < m; i++) {
			dp[i] = triangle.get(n - 1).get(i);
		}

		for (int i = n - 2; i >= 0; i--) {
			for (int j = 0; j <= i; j++) {
				dp[j] = Math.min(dp[j + 1], dp[j]) + triangle.get(i).get(j);
			}
		}
		return dp[0];
	}

	public int uniquePathsWithObstacles(int[][] obstacleGrid) {
		int m = obstacleGrid.length;
		if (m == 0)
			return 0;
		int n = obstacleGrid[0].length;
		int[][] dp = new int[m][n];
		dp[0][0] = obstacleGrid[0][0] == 1 ? 0 : 1;

		for (int i = 1; i < m; i++) {
			dp[i][0] = obstacleGrid[i][0] == 1 ? 0 : dp[i - 1][0];
		}

		for (int i = 1; i < n; i++) {
			dp[0][i] = obstacleGrid[0][i] == 1 ? 0 : dp[0][i - 1];
		}

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				dp[i][j] = obstacleGrid[i][j] == 1 ? 0 : dp[i - 1][j]
						+ dp[i][j - 1];
			}
		}
		return dp[m - 1][n - 1];
	}

	public List<String> restoreIpAddresses(String s) {
		List<String> res = new ArrayList<String>();
		if (s.length() < 4 || s.length() > 12)
			return res;
		restoreIpAddresses(0, s, "", res);
		return res;
	}

	public void restoreIpAddresses(int dep, String s, String sol,
			List<String> res) {
		if (dep == 3 && isValidIpNum(s)) {
			res.add(sol + s);
			return;
		}
		for (int i = 0; i < s.length() && i < 3; i++) {
			String sub = s.substring(0, i + 1);
			if (isValidIpNum(sub)) {
				restoreIpAddresses(dep + 1, s.substring(i + 1),
						sol + sub + ".", res);
			}
		}
	}

	public boolean isValidIpNum(String s) {
		if (s.length() == 0)
			return false;
		if (s.charAt(0) == '0')
			return s.equals("0");
		int num = Integer.parseInt(s);
		return num >= 1 && num <= 255;
	}

	public List<Integer> spiralOrder(int[][] matrix) {
		List<Integer> res = new ArrayList<Integer>();
		int m = matrix.length;
		if (m == 0)
			return res;
		int n = matrix[0].length;
		int top = 0, bottom = m - 1, left = 0, right = n - 1;
		while (true) {
			for (int i = left; i <= right; i++) {
				res.add(matrix[top][i]);
			}
			if (++top > bottom)
				break;

			for (int i = top; i <= bottom; i++) {
				res.add(matrix[i][right]);
			}
			if (--right < left)
				break;

			for (int i = right; i >= left; i--) {
				res.add(matrix[bottom][i]);
			}
			if (--bottom < top)
				break;

			for (int i = bottom; i >= top; i--) {
				res.add(matrix[i][left]);
			}
			if (++left > right)
				break;
		}
		return res;
	}

	public int[][] generateMatrix(int n) {
		int[][] res = new int[n][n];
		int top = 0, bottom = n - 1, left = 0, right = n - 1;
		int num = 1;
		while (true) {
			for (int i = left; i <= right; i++) {
				res[top][i] = num++;
			}
			if (++top > bottom)
				break;

			for (int i = top; i <= bottom; i++) {
				res[i][right] = num++;
			}
			if (--right < left)
				break;

			for (int i = right; i >= left; i--) {
				res[bottom][i] = num++;
			}
			if (--bottom < top)
				break;

			for (int i = bottom; i >= top; i--) {
				res[i][left] = num++;
			}
			if (++left > right)
				break;
		}
		return res;
	}

	public int canCompleteCircuit(int[] gas, int[] cost) {
		int start = 0;
		int sum = 0;
		int cursum = 0;
		for (int i = 0; i < gas.length; i++) {
			sum += gas[i] - cost[i];
			cursum += gas[i] - cost[i];
			if (cursum < 0) {
				cursum = 0;
				start = i + 1;
			}
		}
		return sum >= 0 ? start : -1;
	}

	public boolean canJump(int[] nums) {
		int n = nums.length;
		if (n < 2)
			return true;
		boolean[] dp = new boolean[n];
		dp[n - 1] = true;
		int gap = 1;
		for (int i = n - 2; i >= 0; i--) {
			if (nums[i] >= gap) {
				dp[i] = true;
				gap = 1;
			} else
				gap++;
		}
		return dp[0];
	}

	public boolean canJump2(int[] nums) {
		if (nums.length < 2)
			return true;
		int max = 0;
		for (int i = 0; i < nums.length; i++) {
			max = Math.max(max, i + nums[i]);
			if (max <= i && nums[i] == 0)
				return false;
			if (max >= nums.length - 1)
				return true;
		}
		return false;
	}

	public int jump(int[] nums) {
		if (nums.length < 2)
			return 0;
		int step = 1;
		int max = nums[0];
		int min = 0;
		while (max < nums.length - 1) {
			int t = max;
			for (int i = min; i <= t; i++) {
				if (i + nums[i] > max) {
					max = i + nums[i];
					min = i;
				}
			}
			step++;
		}
		return step;
	}

	public int largestRectangleArea(int[] height) {
		if (height.length == 0)
			return 0;
		int maxArea = 0;
		Stack<Integer> stk = new Stack<Integer>();
		int i = 0;
		while (i < height.length) {
			if (stk.isEmpty() || height[i] >= height[stk.peek()]) {
				stk.push(i++);
			} else {
				int idx = stk.pop();
				int h = height[idx];
				int w = stk.isEmpty() ? i : i - stk.peek() - 1;
				maxArea = Math.max(maxArea, h * w);
			}
		}
		while (!stk.isEmpty()) {
			int idx = stk.pop();
			int h = height[idx];
			int w = stk.isEmpty() ? i : i - stk.peek() - 1;
			maxArea = Math.max(maxArea, h * w);
		}
		return maxArea;
	}

	public int largestRectangleArea2(int[] height) {
		int[] h = Arrays.copyOf(height, height.length + 1);
		int maxArea = 0;
		Stack<Integer> stk = new Stack<Integer>();
		int i = 0;
		while (i < h.length) {
			if (stk.isEmpty() || h[i] >= h[stk.peek()]) {
				stk.push(i++);
			} else {
				int idx = stk.pop();
				int hi = h[idx];
				int w = stk.isEmpty() ? i : i - stk.peek() - 1;
				maxArea = Math.max(maxArea, hi * w);
			}
		}
		return maxArea;
	}

	public int ladderLength(String beginWord, String endWord,
			Set<String> wordList) {
		int len = 1;
		Queue<String> que = new LinkedList<String>();
		int curlevel = 0, nextlevel = 0;
		que.add(beginWord);
		curlevel++;
		wordList.remove(beginWord);

		while (!que.isEmpty()) {
			String word = que.poll();
			curlevel--;
			if (word.equals(endWord))
				return len;
			char[] chars = word.toCharArray();
			for (int i = 0; i < chars.length; i++) {
				char t = chars[i];
				for (char c = 'a'; c <= 'z'; c++) {
					if (c != t) {
						chars[i] = c;
						String s = new String(chars);
						if (wordList.contains(s)) {
							que.add(s);
							wordList.remove(s);
							nextlevel++;
						}
					}
				}
				chars[i] = t;
			}
			if (curlevel == 0) {
				curlevel = nextlevel;
				nextlevel = 0;
				len++;
			}
		}
		return 0;
	}

	public int ladderLength2(String beginWord, String endWord,
			Set<String> wordList) {
		int len = 1;
		Queue<String> que = new LinkedList<String>();
		que.add(beginWord);
		wordList.remove(beginWord);

		while (!que.isEmpty()) {
			int count = que.size();
			for (int i = 0; i < count; i++) {
				String word = que.poll();
				if (word.equals(endWord))
					return len;
				for (char c = 'a'; c <= 'z'; c++) {
					for (int j = 0; j < word.length(); j++) {
						String t = replace(word, j, c);
						if (wordList.contains(t)) {
							que.add(t);
							wordList.remove(t);
						}
					}
				}
			}
			len++;
		}
		return 0;
	}

	public String replace(String s, int j, char c) {
		char[] ch = s.toCharArray();
		ch[j] = c;
		return new String(ch);
	}

	public String longestPalindrome(String s) {
		if (s.length() < 2)
			return s;
		String res = s.substring(0, 1);
		for (int i = 0; i < s.length(); i++) {
			String s1 = expandFromCenter(s, i, i);
			if (s1.length() > res.length())
				res = s1;
			String s2 = expandFromCenter(s, i, i + 1);
			if (s2.length() > res.length())
				res = s2;
		}
		return res;
	}

	public String expandFromCenter(String s, int l, int r) {
		while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
			l--;
			r++;
		}
		return s.substring(l + 1, r);
	}

	public List<String> findMissingRanges(int[] nums, int lower, int upper) {
		List<String> res = new ArrayList<String>();
		if (nums.length == 0) {
			res.add(getRange(lower, upper));
			return res;
		}
		int pre;
		if (nums[0] - lower > 0) {
			res.add(getRange(lower, nums[0] - 1));
			pre = nums[0];
		} else
			pre = lower;
		for (int i = 0; i < nums.length; i++) {
			int cur = nums[i];
			if (cur - pre > 1)
				res.add(getRange(pre + 1, cur - 1));
			pre = cur;
		}
		if (upper - pre > 0)
			res.add(getRange(pre + 1, upper));
		return res;
	}

	public List<String> findMissingRanges2(int[] nums, int lower, int upper) {
		List<String> res = new ArrayList<String>();
		if (nums.length == 0) {
			res.add(getRange(lower, upper));
			return res;
		}
		if (nums[0] > lower)
			res.add(getRange(lower, nums[0] - 1));

		for (int i = 1; i < nums.length; i++) {
			if (nums[i] - nums[i - 1] > 1) {
				res.add(getRange(nums[i - 1] + 1, nums[i] - 1));
			}
		}
		if (nums[nums.length - 1] < upper) {
			res.add(getRange(nums[nums.length - 1] + 1, upper));
		}
		return res;
	}

	public String getRange(int lower, int upper) {
		if (lower == upper)
			return "" + lower;
		return lower + "->" + upper;
	}

	public String shortestPalindrome(String s) {
		int n = s.length();
		if (n < 2)
			return s;
		int i = 0, end = s.length() - 1, j = end;

		while (i < j) {
			if (s.charAt(i) == s.charAt(j)) {
				i++;
				j--;
			} else {
				i = 0;
				end--;
				j = end;
			}
		}
		StringBuilder sb = new StringBuilder(s.substring(end + 1));
		return sb.reverse().append(s).toString();
	}

	public String shortestPalindrome2(String s) {
		int n = s.length();
		if (n < 2)
			return s;
		int mid = n / 2;
		int idx = 0;
		;
		for (int i = mid; i >= 0; i--) {
			if (isValidPalindrome(s, i, 1)) {
				idx = 2 * i + 1;
				break;
			}
			if (isValidPalindrome(s, i, 0)) {
				idx = 2 * i;
				break;
			}
		}
		StringBuilder sb = new StringBuilder(s.substring(idx + 1));
		return sb.reverse().append(s).toString();
	}

	public boolean isValidPalindrome(String s, int center, int shift) {
		int i = center, j = center + shift;
		while (i >= 0 && j < s.length() && s.charAt(i) == s.charAt(j)) {
			i--;
			j++;
		}
		return i < 0;
	}

	public String scanFromCenter(String s, int l, int r) {
		while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) {
			l--;
			r++;
		}
		if (l > 0)
			return null;
		StringBuilder sb = new StringBuilder(s.substring(r));
		sb.reverse();
		return sb.append(s).toString();
	}

	public int maximalSquare(char[][] matrix) {
		int m = matrix.length;
		if (m == 0)
			return 0;
		int n = matrix[0].length;
		int[][] dp = new int[m][n];
		int max = 0;
		for (int i = 0; i < m; i++) {
			dp[i][0] = matrix[i][0] - '0';
			max = Math.max(max, dp[i][0]);
		}

		for (int i = 0; i < n; i++) {
			dp[0][i] = matrix[0][i] - '0';
			max = Math.max(max, dp[0][i]);
		}

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				if (matrix[i][j] == '0')
					dp[i][j] = 0;
				else {
					dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]),
							dp[i - 1][j - 1]) + 1;
					max = Math.max(max, dp[i][j]);
				}
			}
		}
		return max * max;
	}

	public int maximalRectangle(char[][] matrix) {
		int m = matrix.length;
		if (m == 0)
			return 0;
		int n = matrix[0].length;
		int[][] dp = new int[m][n + 1];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (matrix[i][j] == '0')
					dp[i][j] = 0;
				else
					dp[i][j] = i == 0 ? 1 : dp[i - 1][j] + 1;
			}
		}
		int max = 0;
		for (int i = 0; i < m; i++) {
			max = Math.max(max, maxArea(dp[i]));
		}

		return max;
	}

	public int maxArea(int[] height) {
		Stack<Integer> stk = new Stack<Integer>();
		int i = 0;
		int max = 0;
		while (i < height.length) {
			if (stk.isEmpty() || height[stk.peek()] <= height[i])
				stk.push(i++);
			else {
				int top = stk.pop();
				int h = height[top];
				int w = stk.isEmpty() ? i : i - stk.peek() - 1;
				max = Math.max(max, h * w);
			}
		}
		return max;
	}

	public int candy(int[] ratings) {
		int n = ratings.length;
		int[] left = new int[n];
		Arrays.fill(left, 1);
		for (int i = 1; i < n; i++) {
			if (ratings[i] > ratings[i - 1])
				left[i] = left[i - 1] + 1;
		}
		int[] right = new int[n];
		Arrays.fill(right, 1);
		for (int i = n - 2; i >= 0; i--) {
			if (ratings[i] > ratings[i + 1])
				right[i] = right[i + 1] + 1;
		}
		int sum = 0;
		for (int i = 0; i < n; i++) {
			sum += Math.max(left[i], right[i]);
		}
		return sum;
	}

	public int candy2(int[] ratings) {
		int n = ratings.length;
		int[] candies = new int[n];
		Arrays.fill(candies, 1);
		for (int i = 1; i < n; i++) {
			if (ratings[i] > ratings[i - 1])
				candies[i] = candies[i - 1] + 1;
		}
		int sum = candies[n - 1];
		for (int i = n - 2; i >= 0; i--) {
			if (ratings[i] > ratings[i + 1] && candies[i] <= candies[i + 1])
				candies[i] = candies[i + 1] + 1;
			sum += candies[i];
		}
		return sum;
	}

	public boolean canAttendMeetings(Interval[] intervals) {
		if (intervals.length < 2)
			return true;
		Arrays.sort(intervals, new Comparator<Interval>() {
			@Override
			public int compare(Interval i1, Interval i2) {
				return i1.start - i2.start;

			}
		});
		// int end=intervals[0].end;
		for (int i = 1; i < intervals.length; i++) {
			// Interval interval=intervals[i];
			// if(interval.start<end)
			// return false;
			// end=interval.end;
			if (intervals[i].start < intervals[i - 1].end)
				return false;
		}
		return true;
	}

	public int minMeetingRooms(Interval[] intervals) {
		Arrays.sort(intervals, new Comparator<Interval>() {
			@Override
			public int compare(Interval i1, Interval i2) {
				return i1.start - i2.start;

			}
		});

		PriorityQueue<Integer> rooms = new PriorityQueue<Integer>();
		rooms.add(intervals[0].end);

		for (int i = 1; i < intervals.length; i++) {
			if (intervals[i].start >= rooms.peek())
				rooms.poll();
			rooms.offer(intervals[i].end);
		}
		return rooms.size();
	}

	/*
	 * # Very similar with what we do in real life. Whenever you want to start a
	 * meeting, # you go and check if any empty room available (available > 0)
	 * and # if so take one of them ( available -=1 ). Otherwise, # you need to
	 * find a new room someplace else ( numRooms += 1 ). # After you finish the
	 * meeting, the room becomes available again ( available += 1 ).
	 */
	public int minMeetingRooms2(Interval[] intervals) {
		int n = intervals.length;
		int[] starts = new int[n];
		int[] ends = new int[n];
		for (int i = 0; i < n; i++) {
			starts[i] = intervals[i].start;
			ends[i] = intervals[i].end;
		}
		Arrays.sort(starts);
		Arrays.sort(ends);

		int available = 0, rooms = 0, end = 0;
		for (int start : starts) {
			while (start >= ends[end]) {
				available++;
				end++;
			}
			if (available > 0) {
				available--;
			} else
				rooms++;
		}
		return rooms;
	}

	public void wallsAndGates(int[][] rooms) {
		int m = rooms.length;
		if (m == 0)
			return;
		int n = rooms[0].length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (rooms[i][j] == 0)
					bfsSearch(rooms, i * rooms[0].length + j);
			}
		}
	}

	public void bfsSearch(int[][] rooms, int gate) {
		Queue<Integer> que = new LinkedList<Integer>();
		que.add(gate);
		Set<Integer> visited = new HashSet<Integer>();
		int dist = 0;
		while (!que.isEmpty()) {
			int size = que.size();

			for (int i = 0; i < size; i++) {
				int cur = que.poll();
				int row = cur / rooms[0].length;
				int col = cur % rooms[0].length;
				rooms[row][col] = Math.min(dist, rooms[row][col]);
				int up = (row - 1) * rooms[0].length + col;
				int down = (row + 1) * rooms[0].length + col;
				int left = row * rooms[0].length + col - 1;
				int right = row * rooms[0].length + col + 1;

				if (row - 1 >= 0 && rooms[row - 1][col] != -1
						&& !visited.contains(up)) {
					que.add(up);
					visited.add(up);
				}

				if (row + 1 < rooms.length && rooms[row + 1][col] != -1
						&& !visited.contains(down)) {
					que.add(down);
					visited.add(down);
				}

				if (col - 1 >= 0 && rooms[row][col - 1] != -1
						&& !visited.contains(left)) {
					que.add(left);
					visited.add(left);
				}

				if (col + 1 < rooms[0].length && rooms[row][col + 1] != -1
						&& !visited.contains(right)) {
					que.add(right);
					visited.add(right);
				}
			}
			dist++;
		}
	}

	public boolean verifyPreorder(int[] preorder) {
		Stack<Integer> stk = new Stack<Integer>();
		int low = Integer.MIN_VALUE;

		for (int p : preorder) {
			if (p < low)
				return false;
			while (!stk.isEmpty() && p > stk.peek())
				low = stk.pop();
			stk.push(p);
		}
		return true;
	}

	// 0 : 上一轮是0，这一轮过后还是0
	// 1 : 上一轮是1，这一轮过后还是1
	// 2 : 上一轮是1，这一轮过后变为0
	// 3 : 上一轮是0，这一轮过后变为1
	// 这样，对于一个节点来说，如果它周边的点是1或者2，就说明那个点上一轮是活的。最后，在遍历一遍数组，把我们编码再解回去，即0和2都变回0，1和3都变回1，就行了。

	public void gameOfLife(int[][] board) {
		int m = board.length, n = board[0].length;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				int lives = 0;
				// 判断上边
				if (i > 0) {
					lives += board[i - 1][j] == 1 || board[i - 1][j] == 2 ? 1
							: 0;
				}
				// 判断左边
				if (j > 0) {
					lives += board[i][j - 1] == 1 || board[i][j - 1] == 2 ? 1
							: 0;
				}
				// 判断下边
				if (i < m - 1) {
					lives += board[i + 1][j] == 1 || board[i + 1][j] == 2 ? 1
							: 0;
				}
				// 判断右边
				if (j < n - 1) {
					lives += board[i][j + 1] == 1 || board[i][j + 1] == 2 ? 1
							: 0;
				}
				// 判断左上角
				if (i > 0 && j > 0) {
					lives += board[i - 1][j - 1] == 1
							|| board[i - 1][j - 1] == 2 ? 1 : 0;
				}
				// 判断右下角
				if (i < m - 1 && j < n - 1) {
					lives += board[i + 1][j + 1] == 1
							|| board[i + 1][j + 1] == 2 ? 1 : 0;
				}
				// 判断右上角
				if (i > 0 && j < n - 1) {
					lives += board[i - 1][j + 1] == 1
							|| board[i - 1][j + 1] == 2 ? 1 : 0;
				}
				// 判断左下角
				if (i < m - 1 && j > 0) {
					lives += board[i + 1][j - 1] == 1
							|| board[i + 1][j - 1] == 2 ? 1 : 0;
				}
				// 根据周边存活数量更新当前点，结果是0和1的情况不用更新
				if (board[i][j] == 0 && lives == 3) {
					board[i][j] = 3;
				} else if (board[i][j] == 1) {
					if (lives < 2 || lives > 3)
						board[i][j] = 2;
				}
			}
		}
		// 解码
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				board[i][j] = board[i][j] % 2;
			}
		}
	}

	public void gameOfLife2(int[][] board) {
		int m = board.length;
		if (m == 0)
			return;
		int n = board[0].length;
		int[] di = { -1, -1, -1, 0, 0, 1, 1, 1 };
		int[] dj = { -1, 0, 1, -1, 1, -1, 0, 1 };
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				int live = 0;
				for (int k = 0; k < 8; k++) {
					int ii = i + di[k];
					int jj = j + dj[k];
					if (ii < 0 || jj < 0 || ii >= m || jj >= n)
						continue;
					if (board[ii][jj] == 1 || board[ii][jj] == 2)
						live++;
				}
				if (board[i][j] == 0 && live == 3)
					board[i][j] = 3;
				if (board[i][j] == 1 && (live < 2 || live > 3))
					board[i][j] = 2;
			}
		}
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				board[i][j] %= 2;
			}
		}
	}

	public void wiggleSort(int[] nums) {
		Arrays.sort(nums);

		for (int i = 2; i < nums.length; i += 2) {
			int t = nums[i - 1];
			nums[i - 1] = nums[i];
			nums[i] = t;
		}
		System.out.println(Arrays.toString(nums));
	}

	// 如果i是奇数，nums[i] >= nums[i - 1]
	// 如果i是偶数，nums[i] <= nums[i - 1]

	// A[even] <= A[odd],
	// A[odd] >= A[even].
	public void wiggleSortON(int[] nums) {
		for (int i = 1; i < nums.length; i++) {
			if (i % 2 == 0 && nums[i] > nums[i - 1] || i % 2 == 1
					&& nums[i] < nums[i - 1]) {
				int t = nums[i];
				nums[i] = nums[i - 1];
				nums[i - 1] = t;
			}
		}
		System.out.println(Arrays.toString(nums));
	}

	public void wiggleSort2(int[] nums) {
		boolean isLess = true;
		for (int i = 0; i < nums.length - 1; i++) {
			if (isLess) {
				if (nums[i] > nums[i + 1]) {
					swap(nums, i, i + 1);
				}
			} else {
				if (nums[i] < nums[i + 1])
					swap(nums, i, i + 1);
			}
			isLess = !isLess;
		}
		System.out.println(Arrays.toString(nums));
	}

	public int coinChange(int[] coins, int amount) {
		int[] dp = new int[amount + 1];

		for (int i = 1; i <= amount; i++) {
			int min = Integer.MAX_VALUE;
			for (int j = 0; j < coins.length; j++) {
				if (i >= coins[j] && dp[i - coins[j]] != -1)
					min = Math.min(min, dp[i - coins[j]] + 1);
			}
			dp[i] = min == Integer.MAX_VALUE ? -1 : min;
		}
		return dp[amount];
	}

	// follow up:
	// print the coins combination
	/*
	 * Given a total and coins of certain denomination with infinite supply,
	 * what is the minimum number of coins it takes to form this total.
	 * 
	 * Time complexity - O(coins.size * total) Space complexity - O(coins.size *
	 * total)
	 */
	public int coinChangeFollowUp(int[] coins, int amount) {
		int[] dp = new int[amount + 1];
		int[] comb = new int[amount + 1];
		Arrays.fill(comb, -1);
		for (int i = 1; i <= amount; i++) {
			dp[i] = Integer.MAX_VALUE;
			for (int j = 0; j < coins.length; j++) {
				if (i >= coins[j] && dp[i - coins[j]] != -1) {
					if (dp[i] > dp[i - coins[j]] + 1) {
						dp[i] = dp[i - coins[j]] + 1;
						comb[i] = j;
					}
				}
			}
			if (dp[i] == Integer.MAX_VALUE)
				dp[i] = -1;
		}
		System.out.println(Arrays.toString(dp));
		printCoinsCombination(comb, coins);
		return dp[amount];
	}

	public void printCoinsCombination(int[] comb, int[] coins) {
		if (comb[comb.length - 1] == -1) {
			System.out.println("no solution!");
			return;
		}
		int start = comb.length - 1;
		while (start > 0) {
			int j = comb[start];
			System.out.print(coins[j] + " ");
			start -= coins[j];
		}
		System.out.println();
	}

	public String removeDup(String s) {
		if (s.length() < 2)
			return s;
		Stack<Character> stk = new Stack<Character>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (stk.isEmpty() || c != stk.peek())
				stk.push(c);
			else {
				while (i < s.length() && s.charAt(i) == stk.peek())
					i++;
				i--;
				stk.pop();
			}
		}
		StringBuilder sb = new StringBuilder();
		while (!stk.isEmpty()) {
			sb.append(stk.pop());
		}
		return sb.reverse().toString();
	}

	public int bulbSwitch(int n) {
		// boolean[] bulbs=new boolean[n];

		// for(int i=0;i<n;i++){
		// bulbs[i]=true;
		// }
		// int d=2;
		// for(int i=0;i<n;i++){
		// for(int j=0;j<n;j++){
		// if(j%d==0)
		// bulbs[j]=!bulbs[j];
		// }
		// d++;
		// }

		// int count=0;
		// for(int i=0;i<n;i++){
		// if(bulbs[i])
		// count++;
		// }
		// return count;

		return (int) Math.sqrt(n);
	}

	public int minTotalDistance(int[][] grid) {
		List<Integer> ipos = new ArrayList<Integer>();
		List<Integer> jpos = new ArrayList<Integer>();
		// 统计出有哪些横纵坐标
		for (int i = 0; i < grid.length; i++) {
			for (int j = 0; j < grid[0].length; j++) {
				if (grid[i][j] == 1) {
					ipos.add(i);
					jpos.add(j);
				}
			}
		}
		int sum = 0;
		// 计算纵坐标到纵坐标中点的距离，这里不需要排序，因为之前统计时是按照i的顺序
		for (Integer pos : ipos) {
			sum += Math.abs(pos - ipos.get(ipos.size() / 2));
		}
		// 计算横坐标到横坐标中点的距离，这里需要排序，因为统计不是按照j的顺序
		Collections.sort(jpos);
		for (Integer pos : jpos) {
			sum += Math.abs(pos - jpos.get(jpos.size() / 2));
		}
		return sum;
	}

	// meeting point cannot be at one of the k points.
	public int minTotalDistance2(int[][] grid) {
		List<Integer> ipos = new ArrayList<Integer>();
		List<Integer> jpos = new ArrayList<Integer>();
		// 统计出有哪些横纵坐标
		for (int i = 0; i < grid.length; i++) {
			for (int j = 0; j < grid[0].length; j++) {
				if (grid[i][j] == 1) {
					ipos.add(i);
					jpos.add(j);
				}
			}
		}

		// 计算纵坐标到纵坐标中点的距离，这里不需要排序，因为之前统计时是按照i的顺序
		int tx = ipos.get(ipos.size() / 2);
		// 计算横坐标到横坐标中点的距离，这里需要排序，因为统计不是按照j的顺序
		Collections.sort(jpos);
		int ty = jpos.get(jpos.size() / 2);
		System.out.println("tx is " + tx + ", ty is " + ty);
		if (grid[tx][ty] != 1) {
			return totalDist(tx, ty, ipos, jpos);
		} else {
			PriorityQueue<MeetingPoint> heap = new PriorityQueue<MeetingPoint>(
					4, new Comparator<MeetingPoint>() {
						@Override
						public int compare(MeetingPoint p1, MeetingPoint p2) {
							return p1.dist - p2.dist;
						}
					});
			MeetingPoint mp = new MeetingPoint(tx, ty);
			mp.dist = Integer.MAX_VALUE;
			heap.offer(mp);
			while (!heap.isEmpty()) {
				MeetingPoint p = heap.poll();
				System.out.println("px is " + p.x + ", py is " + p.y);
				if (grid[p.x][p.y] != 1)
					return totalDist(p.x, p.y, ipos, jpos);
				if (p.x > 0) {
					MeetingPoint tp = new MeetingPoint(p.x - 1, p.y);
					tp.dist = Math.abs(tx - (p.x - 1)) + Math.abs(ty - p.y);
					heap.offer(tp);
				}
				if (p.y > 0) {
					MeetingPoint tp = new MeetingPoint(p.x, p.y - 1);
					tp.dist = Math.abs(tx - p.x) + Math.abs(ty - (p.y - 1));
					heap.offer(tp);
				}
				if (p.x < grid.length - 1) {
					MeetingPoint tp = new MeetingPoint(p.x + 1, p.y);
					tp.dist = Math.abs(tx - (p.x + 1)) + Math.abs(ty - p.y);
					heap.offer(tp);
				}
				if (p.y < grid[0].length - 1) {
					MeetingPoint tp = new MeetingPoint(p.x, p.y + 1);
					tp.dist = Math.abs(tx - p.x) + Math.abs(ty - (p.y + 1));
					heap.offer(tp);
				}
			}
			return -1;
		}

	}

	public int totalDist(int x, int y, List<Integer> ipos, List<Integer> jpos) {
		int sum = 0;
		for (int i = 0; i < ipos.size(); i++) {
			sum += Math.abs(ipos.get(i) - x);
		}
		for (int j = 0; j < jpos.size(); j++) {
			sum += Math.abs(jpos.get(j) - y);
		}
		return sum;
	}

	public List<List<Integer>> levelOrderBottom(TreeNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (root == null)
			return res;
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		que.add(root);
		while (!que.isEmpty()) {
			int size = que.size();
			List<Integer> level = new ArrayList<Integer>();
			for (int i = 0; i < size; i++) {
				TreeNode top = que.remove();
				level.add(top.val);
				if (top.left != null)
					que.add(top.left);
				if (top.right != null)
					que.add(top.right);
			}
			res.add(level);
		}
		Collections.reverse(res);
		return res;
	}

	public List<String> removeInvalidParentheses(String s) {
		List<String> res = new ArrayList<String>();
		Queue<String> que = new LinkedList<String>();
		Set<String> visited = new HashSet<String>();
		que.add(s);
		visited.add(s);
		boolean found = false;
		while (!que.isEmpty()) {
			String candidate = que.poll();
			System.out.println(candidate);
			if (isValidParen(candidate)) {
				res.add(candidate);
				found = true;
			}
			if (!found) {
				for (int i = 0; i < candidate.length(); i++) {
					char c = candidate.charAt(i);
					if (c == '(' || c == ')') {
						String next = candidate.substring(0, i)
								+ candidate.substring(i + 1);

						if (!visited.contains(next)) {
							que.add(next);
							visited.add(next);
							System.out.println("next " + next);
						}
					}
				}
			}
		}
		return res;
	}

	public boolean isValidParen(String s) {
		int open = 0;
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == '(')
				open++;
			else if (s.charAt(i) == ')') {
				if (--open < 0)
					return false;
			}
		}
		return open == 0;
	}

	// Suppose there are N elements and they range from A to B.
	//
	// Then the maximum gap will be no smaller than ceiling[(B - A) / (N - 1)]
	//
	// Let the length of a bucket to be len = ceiling[(B - A) / (N - 1)], then
	// we will have at most num = (B - A) / len + 1 of bucket
	//
	// for any number K in the array, we can easily find out which bucket it
	// belongs by calculating loc = (K - A) / len and therefore maintain the
	// maximum and minimum elements in each bucket.
	//
	// Since the maximum difference between elements in the same buckets will be
	// at most len - 1, so the final answer will not be taken from two elements
	// in the same buckets.
	public int maximumGap(int[] nums) {
		int n = nums.length;
		if (n < 2)
			return 0;
		int max = Integer.MIN_VALUE, min = Integer.MAX_VALUE;
		for (int i = 0; i < n; i++) {
			max = Math.max(max, nums[i]);
			min = Math.min(min, nums[i]);
		}

		int gap = (int) Math.ceil((double) (max - min) / (n - 1));
		// int bucketNum=(int)Math.ceil((double)(max-min)/gap+1);

		int[] bucketMin = new int[n - 1];
		int[] bucketMax = new int[n - 1];
		Arrays.fill(bucketMin, Integer.MAX_VALUE);
		Arrays.fill(bucketMax, Integer.MIN_VALUE);
		for (int i = 0; i < n; i++) {
			if (nums[i] == min || nums[i] == max)
				continue;
			int idx = (nums[i] - min) / gap;
			bucketMin[idx] = Math.min(nums[i], bucketMin[idx]);
			bucketMax[idx] = Math.max(nums[i], bucketMax[idx]);
		}
		int maxGap = Integer.MIN_VALUE;
		int pre = min;
		for (int i = 0; i < n - 1; i++) {
			if (bucketMin[i] == Integer.MAX_VALUE
					|| bucketMax[i] == Integer.MIN_VALUE)
				continue;
			maxGap = Math.max(maxGap, bucketMin[i] - pre);
			pre = bucketMax[i];
		}
		maxGap = Math.max(maxGap, max - pre);
		return maxGap;
	}

	public int numDistinct(String s, String t) {
		if (s.length() < t.length())
			return 0;
		int m = s.length(), n = t.length();
		int[][] dp = new int[m + 1][n + 1];

		for (int i = 0; i < m; i++) {
			dp[i][0] = 1;
		}
		for (int i = 1; i <= m; i++) {
			for (int j = 1; j <= n; j++) {
				if (s.charAt(i - 1) == t.charAt(j - 1))
					dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
				else
					dp[i][j] = dp[i - 1][j];
			}
		}
		return dp[m][n];
	}

	public int countPrimes(int n) {
		int count = 0;
		for (int i = 1; i < n; i++) {
			if (isPrime(i))
				count++;
		}
		return count;
	}

	public boolean isPrime(int num) {
		if (num < 2)
			return false;
		// Loop's ending condition is i * i <= num instead of i <= sqrt(num)
		// to avoid repeatedly calling an expensive function sqrt().
		for (int i = 2; i * i <= num; i++) {
			if (num % i == 0)
				return false;
		}
		return true;

	}

	// Sieve of Eratosthenes
	public int countPrimes2(int n) {
		if (n < 3)
			return 0;

		boolean[] primes = new boolean[n];
		for (int i = 0; i < n; i++) {
			primes[i] = true;
		}

		for (int i = 2; i < Math.sqrt(n) + 1; i++) {
			if (primes[i]) {
				for (int j = i * i; j < n; j += i) {
					primes[j] = false;
				}
			}
		}
		int count = 0;
		for (int i = 2; i < n; i++) {
			if (primes[i])
				count++;
		}
		return count;
	}

	public boolean isInterleave(String s1, String s2, String s3) {
		if (s1.length() + s2.length() != s3.length())
			return false;
		int n1 = s1.length(), n2 = s2.length();
		boolean[][] dp = new boolean[n1 + 1][n2 + 1];
		dp[0][0] = true;
		for (int i = 1; i <= n1; i++) {
			dp[i][0] = s1.charAt(i - 1) == s3.charAt(i - 1) && dp[i - 1][0];
		}
		for (int i = 1; i <= n2; i++) {
			dp[0][i] = s2.charAt(i - 1) == s3.charAt(i - 1) && dp[0][i - 1];
		}

		for (int i = 1; i <= n1; i++) {
			for (int j = 1; j <= n2; j++) {
				dp[i][j] = s1.charAt(i - 1) == s3.charAt(i + j - 1)
						&& dp[i - 1][j]
						|| s2.charAt(j - 1) == s3.charAt(i + j - 1)
						&& dp[i][j - 1];
			}
		}
		return dp[n1][n2];
	}

	public int numTrees(int n) {
		if (n < 2)
			return 1;
		int total = 0;
		for (int i = 1; i <= n; i++) {
			int left = numTrees(i - 1);
			int right = numTrees(n - i);
			total += left * right;
		}
		return total;
	}

	// 定义Count[i] 为以[0,i]能产生的Unique Binary Tree的数目，
	public int numTrees2dp(int n) {
		if (n < 2)
			return 1;
		int[] dp = new int[n + 1];
		dp[0] = dp[1] = 1;

		for (int i = 2; i <= n; i++) {
			for (int j = 0; j <= i - 1; j++) {
				dp[i] += dp[j] * dp[i - 1 - j];
			}
		}
		return dp[n];
	}

	public List<TreeNode> generateTrees(int n) {
		if (n < 1)
			return new ArrayList<TreeNode>();
		return generateTrees(1, n);
	}

	public List<TreeNode> generateTrees(int beg, int end) {
		List<TreeNode> res = new ArrayList<TreeNode>();
		if (beg > end) {
			res.add(null);
			return res;
		}

		for (int i = beg; i <= end; i++) {
			List<TreeNode> left = generateTrees(beg, i - 1);
			List<TreeNode> right = generateTrees(i + 1, end);

			for (int j = 0; j < left.size(); j++) {
				for (int k = 0; k < right.size(); k++) {
					TreeNode root = new TreeNode(i);
					root.left = left.get(j);
					root.right = right.get(k);
					res.add(root);
				}
			}
		}
		return res;
	}

	// rotate duplicates
	public boolean search2(int[] nums, int target) {
		int beg = 0, end = nums.length - 1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (nums[mid] == target)
				return true;
			if (nums[beg] < nums[mid]) {
				if (nums[beg] <= target && target < nums[mid])
					end = mid - 1;
				else
					beg = mid + 1;
			} else if (nums[mid] < nums[beg]) {
				if (nums[mid] < target && target <= nums[end])
					beg = mid + 1;
				else
					end = mid - 1;
			} else
				beg++;
		}
		return false;
	}

	public String getHint(String secret, String guess) {
		// HashMap<Character, Integer> map=new HashMap<Character, Integer>();
		// int cows=0, bulls=0;
		// for(int i=0;i<secret.length();i++){
		// char c=secret.charAt(i);
		// if(c==guess.charAt(i))
		// bulls++;
		// else{
		// if(map.containsKey(c))
		// map.put(c, map.get(c)+1);
		// else
		// map.put(c, 1);
		// }
		// }
		//
		// for(int i=0;i<secret.length();i++){
		// char c=guess.charAt(i);
		// if(secret.charAt(i)!=c&&map.containsKey(c)){
		// cows++;
		// if(map.get(c)==1)
		// map.remove(c);
		// else
		// map.put(c, map.get(c)-1);
		// }
		// }
		// return bulls+"A"+cows+"B";
		int[] map = new int[256];
		int cows = 0, bulls = 0;
		for (int i = 0; i < secret.length(); i++) {
			char c = secret.charAt(i);
			if (c == guess.charAt(i))
				bulls++;
			else {
				map[c]++;
			}
		}

		for (int i = 0; i < secret.length(); i++) {
			char c = guess.charAt(i);
			if (secret.charAt(i) != c && map[c] > 0) {
				cows++;
				map[c]--;
			}
		}
		return bulls + "A" + cows + "B";
	}

	/*
	 * 在处理不是bulls的位置时，我们看如果secret当前位置数字的映射值小于0，则表示其在guess中出现过，cows自增1，然后映射值加1，
	 * 如果guess当前位置的数字的映射值大于0，则表示其在secret中出现过，cows自增1，然后映射值减1
	 */

	public String getHint2(String secret, String guess) {
		int[] map = new int[256];
		int cows = 0, bulls = 0;

		for (int i = 0; i < secret.length(); i++) {
			char c1 = secret.charAt(i);
			char c2 = guess.charAt(i);
			if (c1 == c2)
				bulls++;
			else {
				if (map[c1]++ < 0)
					cows++;
				if (map[c2]-- > 0)
					cows++;
			}
		}
		return bulls + "A" + cows + "B";
	}

	public int[][] multiply(int[][] A, int[][] B) {
		int r1 = A.length, c1 = A[0].length, c2 = B[0].length;
		int[][] C = new int[r1][c2];

		for (int i = 0; i < r1; i++) {
			for (int j = 0; j < c1; j++) {
				if (A[i][j] != 0) {
					for (int k = 0; k < c2; k++) {
						if (B[j][k] != 0)
							C[i][k] += A[i][j] * B[j][k];
					}
				}
			}
		}
		for (int i = 0; i < r1; i++) {
			System.out.println(Arrays.toString(C[i]));
		}
		return C;
	}

	public int[][] multiply2(int[][] A, int[][] B) {
		int r1 = A.length, c1 = A[0].length, c2 = B[0].length;
		int[][] C = new int[r1][c2];
		List[] indexA = new List[r1];

		for (int i = 0; i < r1; i++) {
			List<Integer> lst = new ArrayList<Integer>();
			for (int j = 0; j < c1; j++) {
				if (A[i][j] != 0) {
					lst.add(j);
					lst.add(A[i][j]);
				}
			}
			indexA[i] = lst;
		}

		for (int i = 0; i < r1; i++) {
			List<Integer> numsA = indexA[i];
			for (int j = 0; j < numsA.size() - 1; j += 2) {
				int index = numsA.get(j);
				int num = numsA.get(j + 1);
				for (int k = 0; k < c2; k++) {
					if (B[index][k] != 0)
						C[i][k] += num * B[index][k];
				}
			}
		}
		for (int i = 0; i < r1; i++) {
			System.out.println(Arrays.toString(C[i]));
		}

		return C;
	}

	// Longest Span with same Sum in two Binary arrays
	public int longestCommonSum(int[] A, int[] B) {
		int n = A.length;
		int[] diffs = new int[2 * n + 1];
		for (int i = 0; i < 2 * n + 1; i++)
			diffs[i] = -1;

		int maxLen = 0;
		int preSumA = 0, preSumB = 0;
		for (int i = 0; i < n; i++) {
			preSumA += A[i];
			preSumB += B[i];

			// Comput current diff and index to be used
			// in diff array. Note that diff can be negative
			// and can have minimum value as -n.
			int curDiff = preSumA - preSumB;
			int diffIndex = n + curDiff;

			if (curDiff == 0)
				maxLen = i + 1;
			if (diffs[diffIndex] == -1)
				diffs[diffIndex] = i;
			else {
				int len = i - diffs[diffIndex];
				maxLen = Math.max(maxLen, len);
			}
		}
		return maxLen;
	}

	public List<String> wordAbbreviation(String word) {
		List<String> res = new ArrayList<String>();
		int len = word.length();
		if (len == 0)
			return res;
		StringBuilder sb = new StringBuilder();

		for (int i = 1; i < len; i++) {
			sb.append(word.charAt(0));
			sb.append(len - i);
			sb.append(word.substring(len - i + 1));
			res.add(sb.toString());
			sb = new StringBuilder();
		}
		res.add("" + len);
		return res;
	}

	public int lengthOfLongestSubstringTwoDistinct2(String s) {
		if (s.length() < 3)
			return s.length();
		int start = 0;
		int maxLen = 0;
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (!map.containsKey(c)) {
				if (map.size() < 2)
					map.put(c, 1);
				else {
					while (map.size() == 2) {
						char ch = s.charAt(start);
						if (map.get(ch) == 1)
							map.remove(ch);
						else
							map.put(ch, map.get(ch) - 1);
						start++;
					}
					map.put(c, 1);
				}
			} else {
				map.put(c, map.get(c) + 1);
				maxLen = Math.max(maxLen, i - start + 1);
			}
		}
		return maxLen;
	}

	public boolean isAdditiveNumber(String num) {
		for (int i = 1; i < num.length() - 1; i++) {
			for (int j = i + 1; j < num.length(); j++) {
				if (dfsAdditiveNumber(num, 0, i, j))
					return true;
			}
		}
		return false;
	}

	public boolean dfsAdditiveNumber(String num, int start, int end1, int end2) {
		String s1 = num.substring(start, end1);
		String s2 = num.substring(end1, end2);
		if (s1.length() > 1 && s1.charAt(0) == '0' || s2.length() > 1
				&& s2.charAt(0) == '0')
			return false;
		long num1 = Long.valueOf(s1);
		long num2 = Long.valueOf(s2);
		String sum = String.valueOf(num1 + num2);
		if (num.substring(end2).length() == 0)
			return true;
		if (num.substring(end2).indexOf(sum) != 0)
			return false;
		return dfsAdditiveNumber(num, end1, end2, end2 + sum.length());
	}

	public boolean isAdditiveNumber2(String num) {
		int len = num.length();
		for (int i = 1; i < len / 2; i++) {
			if (num.charAt(0) == '0' && i >= 2)
				continue;
			for (int j = i + 1; len - j >= i && len - j >= j - i; j++) {
				if (num.charAt(i) == '0' && j >= i + 2)
					continue;
				long num1 = Long.parseLong(num.substring(0, i));
				long num2 = Long.parseLong(num.substring(i, j));
				String sub = num.substring(j);
				if (dfsAdditiveNumber(sub, num1, num2))
					return true;
			}
		}
		return false;
	}

	public boolean dfsAdditiveNumber(String str, long num1, long num2) {
		if (str.length() == 0)
			return true;
		long sum = num1 + num2;
		String s = String.valueOf(sum);

		if (!str.startsWith(s))
			return false;
		return dfsAdditiveNumber(str.substring(s.length()), num2, sum);
	}

	public int maxSubArrayLen(int[] nums, int k) {
		int maxLen = 0;
		int sum = 0;
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		map.put(0, -1);
		for (int i = 0; i < nums.length; i++) {
			sum += nums[i];
			if (!map.containsKey(sum))
				map.put(sum, i);
			if (map.containsKey(sum - k)) {
				maxLen = Math.max(maxLen, i - map.get(sum - k));
			}
		}
		return maxLen;
	}

	// goole interview construct tree from inpur file
	/*
	 * 给n行输入，每行2个int： 1 2 1 3 1 4 2 5 2 6 3 7 ... 构建一棵树（不一定是binary
	 * tree），第一行输入的第一个int是root，所以上面的输入转换成下面的树： 1 / | \ 2 3 4 / \ | 5 6 7
	 */

	public Node buildTree(List<Pair> list) {
		if (list.size() == 0)
			return null;
		Node root = new Node(list.get(0).p);
		Queue<Node> que = new LinkedList<Node>();
		que.add(root);
		for (int i = 0; i < list.size(); i++) {
			Node parent = que.peek();
			Pair cur = list.get(i);
			while (!que.isEmpty() && cur.p != parent.val) {
				que.remove();
				parent = que.peek();
			}

			Node child = new Node(cur.c);
			parent.children.add(child);
			que.add(child);
		}
		return root;
	}

	public List<List<Character>> printTree(Node root) {
		List<List<Character>> res = new ArrayList<List<Character>>();
		if (root == null)
			return res;
		Queue<Node> que = new LinkedList<Node>();
		List<Character> level = new ArrayList<Character>();
		int curlevel = 0, nextlevel = 0;
		que.add(root);
		curlevel++;
		while (!que.isEmpty()) {
			Node top = que.remove();
			level.add(top.val);
			curlevel--;
			for (Node node : top.children) {
				que.add(node);
				nextlevel++;
			}
			if (curlevel == 0) {
				curlevel = nextlevel;
				nextlevel = 0;
				res.add(level);
				level = new ArrayList<Character>();
			}
		}
		return res;
	}

	public int trailingZeroes(int n) {
		int count = 0;
		while (n > 0) {
			count += n / 5;
			n /= 5;
		}
		return count;
	}

	public int trailingZeroes2(int n) {
		int count = 0;
		int x = 5;
		while (n >= x) {
			count += n / x;
			x *= 5;
		}
		return count;
	}

	public List<List<String>> partition(String s) {
		List<List<String>> res = new ArrayList<List<String>>();
		List<String> sol = new ArrayList<String>();
		partitionUtil(s, 0, sol, res);
		return res;
	}

	public void partitionUtil(String s, int cur, List<String> sol,
			List<List<String>> res) {
		if (cur == s.length()) {
			res.add(new ArrayList<String>(sol));
			return;
		}
		for (int i = cur; i < s.length(); i++) {
			String sub = s.substring(cur, i + 1);
			if (isPalindrome(sub)) {
				sol.add(sub);
				partitionUtil(s, i + 1, sol, res);
				sol.remove(sol.size() - 1);
			}
		}
	}

	public boolean isPalindrome(String s) {
		if (s.length() < 2)
			return true;
		int beg = 0, end = s.length() - 1;
		while (beg < end) {
			if (s.charAt(beg++) != s.charAt(end--))
				return false;
		}
		return true;
	}

	public boolean isPalindrome(int x) {
		if (x < 0 || x != 0 && x % 10 == 0)
			return false;
		int res = 0, t = x;
		while (x > 0) {
			res = res * 10 + x % 10;
			x /= 10;
		}
		return res == t;
	}

	// public List<List<String>> partition2(String s) {
	// List<List<String>> res=new ArrayList<List<String>>();
	// int n=s.length();
	// boolean[][] dp=new boolean[n][n];
	//
	// for(int l=1;l<=n;l++){
	// List<String> sol=new ArrayList<String>();
	// for(int i=0;i<n-l+1;i++){
	// int j=i+l-1;
	// if(s.charAt(i)==s.charAt(j)&&(j-i<=1||dp[i+1][j-1])){
	// dp[i][j]=true;
	// sol.add(s.substring(i, j+1));
	// }
	// }
	// if(sol.size()>0)
	// res.add(sol);
	// }
	// return res;
	// }

	// cuts[i] means min cut from i to the end
	public int minCut(String s) {
		int n = s.length();
		int[] cuts = new int[n + 1];
		boolean[][] dp = new boolean[n][n];
		for (int i = 0; i <= n; i++) {
			cuts[i] = n - i;
		}

		for (int i = n - 1; i >= 0; i--) {
			for (int j = i; j < n; j++) {
				if (s.charAt(i) == s.charAt(j)
						&& (j - i <= 1 || dp[i + 1][j - 1])) {
					dp[i][j] = true;
					cuts[i] = Math.min(cuts[i], cuts[j + 1] + 1);
				}
			}
		}
		return cuts[0] - 1;
	}

	public int countRangeSum(int[] nums, int lower, int upper) {
		int n = nums.length;
		int[] sum = new int[n + 1];
		for (int i = 1; i <= nums.length; i++) {
			sum[i] = sum[i - 1] + nums[i - 1];
		}

		int count = 0;
		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j <= n; j++) {
				if (sum[j] - sum[i] >= lower && sum[j] - sum[i] <= upper)
					count++;
			}
		}
		return count;
	}

	public boolean wordPatternMatch(String pattern, String str) {
		Map<Character, String> map = new HashMap<Character, String>();
		Set<String> set = new HashSet<String>();
		return dfsMatch(str, 0, pattern, 0, map, set);
	}

	public boolean dfsMatch(String str, int i, String pattern, int j,
			Map<Character, String> map, Set<String> set) {
		if (i == str.length() && j == pattern.length())
			return true;
		if (i == str.length() || j == pattern.length())
			return false;
		char c = pattern.charAt(j);
		if (map.containsKey(c)) {
			String s = map.get(c);
			if (!str.startsWith(s, i))
				return false;
			return dfsMatch(str, i + s.length(), pattern, j + 1, map, set);
		}

		for (int k = i; k < str.length(); k++) {
			String s = str.substring(i, k + 1);
			if (set.contains(s))
				continue;
			map.put(c, s);
			set.add(s);

			if (dfsMatch(str, k + 1, pattern, j + 1, map, set))
				return true;
			map.remove(c);
			set.remove(s);
		}
		return false;
	}

	public String convert2(String s, int numRows) {
		if (numRows == 1 || s.length() < numRows)
			return s;
		StringBuilder sb = new StringBuilder();
		int zigSize = 2 * numRows - 2;

		for (int i = 0; i < numRows; i++) {
			for (int j = i; j < s.length(); j += zigSize) {
				sb.append(s.charAt(j));
				if (i > 0 && i < numRows - 1) {
					int tmp = j + zigSize - 2 * i;
					if (tmp < s.length()) {
						sb.append(s.charAt(tmp));
					}
				}
			}
		}
		return sb.toString();
	}

	public String convert(String s, int numRows) {
		if (numRows == 1 || s.length() <= numRows)
			return s;
		StringBuilder[] rows = new StringBuilder[numRows];
		boolean down = true;
		int row = 0;
		for (int i = 0; i < s.length(); i++) {
			if (rows[row] == null)
				rows[row] = new StringBuilder();
			rows[row].append(s.charAt(i));
			if (row == numRows - 1)
				down = false;
			else if (row == 0)
				down = true;
			if (down)
				row++;
			else
				row--;
		}
		String res = "";
		for (int i = 0; i < numRows; i++) {
			res += rows[i];
		}
		return res;
	}

	public String convert3(String s, int numRows) {
		if (numRows == 1 || s.length() <= numRows)
			return s;
		StringBuilder[] rows = new StringBuilder[numRows];
		int flag = -1;
		int row = 0;
		for (int i = 0; i < s.length(); i++) {
			if (rows[row] == null)
				rows[row] = new StringBuilder();
			rows[row].append(s.charAt(i));
			if (row == numRows - 1)
				flag = 1;
			else if (row == 0)
				flag = -1;
			row += flag;
		}
		String res = "";
		for (int i = 0; i < numRows; i++) {
			res += rows[i];
		}
		return res;
	}

	public List<List<String>> groupStrings(String[] strings) {
		List<List<String>> res = new ArrayList<List<String>>();
		Map<String, List<String>> map = new HashMap<String, List<String>>();

		for (String s : strings) {
			String key = getMasks(s);
			if (!map.containsKey(key))
				map.put(key, new ArrayList<String>());
			map.get(key).add(s);
		}

		for (List<String> lst : map.values()) {
			Collections.sort(lst);
			res.add(lst);
		}
		return res;
	}

	public String getMasks(String s) {
		StringBuilder sb = new StringBuilder();
		for (int i = 1; i < s.length(); i++) {
			sb.append((s.charAt(i) - s.charAt(i - 1) + 26) % 26);
		}
		return sb.toString();
	}

	// flip game
	public List<String> generatePossibleNextMoves(String s) {
		List<String> res = new ArrayList<String>();
		for (int i = 0; i < s.length() - 1; i++) {
			if (s.charAt(i) == '+' && s.charAt(i + 1) == '+') {
				String temp = s.substring(0, i) + "--" + s.substring(i + 2);
				res.add(temp);
			}
		}
		return res;
	}

	public boolean canWin(String s) {
		for (int i = 0; i < s.length() - 1; i++) {
			if (s.charAt(i) == '+' && s.charAt(i + 1) == '+'
					&& !canWin(s.substring(0, i) + "--" + s.substring(i + 2)))
				return true;
		}
		return false;
	}

	public boolean canWin2(String s) {
		return canWin(s.toCharArray());
	}

	public boolean canWin(char[] s) {
		for (int i = 0; i < s.length; i++) {
			if (s[i] == '+' && s[i + 1] == '+') {
				s[i] = s[i + 1] = '-';
				if (!canWin(s))
					return true;
				s[i] = s[i + 1] = '+';
			}
		}
		return false;
	}

	// For each node there can be four ways that the max path goes through the
	// node:
	// 1. Node only
	// 2. Max path through Left Child + Node
	// 3. Max path through Right Child + Node
	// 4. Max path through Left Child + Node + Max path through Right Child

	public int maxPathSum(TreeNode root) {
		int[] max = { Integer.MIN_VALUE };
		maxPathSum(root, max);
		return max[0];
	}

	public int maxPathSum(TreeNode root, int[] max) {
		if (root == null)
			return 0;
		int leftMax = maxPathSum(root.left, max);
		int rightMax = maxPathSum(root.right, max);
		int tMax = Math.max(root.val,
				Math.max(leftMax + root.val, rightMax + root.val));
		max[0] = Math
				.max(max[0], Math.max(tMax, leftMax + root.val + rightMax));
		return tMax;
	}

	public int closestValue(TreeNode root, double target) {
		int closest = root.val;
		double minDif = Double.MAX_VALUE;
		TreeNode cur = root;
		while (cur != null) {
			if (Math.abs(cur.val - target) < minDif) {
				minDif = Math.abs(cur.val - target);
				closest = cur.val;
			}
			if (cur.val == target)
				return cur.val;
			if (cur.val < target)
				cur = cur.right;
			else
				cur = cur.left;
		}
		return closest;
	}

	public double findMedianSortedArrays(int[] nums1, int[] nums2) {
		int m = nums1.length;
		int n = nums2.length;

		if ((m + n) % 2 == 0) {
			return (findKth(nums1, 0, m, nums2, 0, n, (m + n) / 2) + findKth(
					nums1, 0, m, nums2, 0, n, (m + n) / 2 + 1)) / 2.0;
		}
		return findKth(nums1, 0, m, nums2, 0, n, (m + n) / 2 + 1);
	}

	public int findKth(int[] nums1, int aoffset, int m, int[] nums2,
			int boffset, int n, int k) {
		if (m > n)
			return findKth(nums2, boffset, n, nums1, aoffset, m, k);
		if (m == 0)
			return nums2[boffset + k - 1];
		if (k == 1)
			return Math.min(nums1[aoffset], nums2[boffset]);
		int pa = Math.min(m, k / 2);
		int pb = k - pa;
		if (nums1[aoffset + pa - 1] < nums2[boffset + pb - 1])
			return findKth(nums1, aoffset + pa, m - pa, nums2, boffset, n, k
					- pa);
		else
			return findKth(nums1, aoffset, m, nums2, boffset + pb, n - pb, k
					- pb);
	}

	/*
	 * if (aMid < bMid) Keep [aRight + bLeft] else Keep [bRight + aLeft]
	 */

	public double findMedianSortedArrays2(int[] nums1, int[] nums2) {
		int m = nums1.length, n = nums2.length;
		int l = (m + n + 1) / 2;
		int r = (m + n + 2) / 2;
		return (findKthNumber(nums1, 0, nums2, 0, l) + findKthNumber(nums1, 0,
				nums2, 0, r)) / 2.0;
	}

	public int findKthNumber(int[] nums1, int aoffset, int[] nums2,
			int boffset, int k) {
		if (aoffset > nums1.length - 1)
			return nums2[boffset + k - 1];
		if (boffset > nums2.length - 1)
			return nums1[aoffset + k - 1];
		if (k == 1)
			return Math.min(nums1[aoffset], nums2[boffset]);
		int aMid = Integer.MAX_VALUE, bMid = Integer.MAX_VALUE;
		if (aoffset + k / 2 - 1 < nums1.length)
			aMid = nums1[aoffset + k / 2 - 1];
		if (boffset + k / 2 - 1 < nums2.length)
			bMid = nums2[boffset + k / 2 - 1];

		if (aMid < bMid)
			return findKthNumber(nums1, aoffset + k / 2, nums2, boffset, k - k
					/ 2);
		return findKthNumber(nums1, aoffset, nums2, boffset + k / 2, k - k / 2);
	}

	public double findMedianSortedArrays3(int[] A, int[] B) {
		int m = A.length, n = B.length;
		int l = (m + n + 1) / 2;
		int r = (m + n + 2) / 2;
		return (getkth(A, 0, B, 0, l) + getkth(A, 0, B, 0, r)) / 2.0;
	}

	public double getkth(int[] A, int aStart, int[] B, int bStart, int k) {
		if (aStart > A.length - 1)
			return B[bStart + k - 1];
		if (bStart > B.length - 1)
			return A[aStart + k - 1];
		if (k == 1)
			return Math.min(A[aStart], B[bStart]);

		int aMid = Integer.MAX_VALUE, bMid = Integer.MAX_VALUE;
		if (aStart + k / 2 - 1 < A.length)
			aMid = A[aStart + k / 2 - 1];
		if (bStart + k / 2 - 1 < B.length)
			bMid = B[bStart + k / 2 - 1];

		if (aMid < bMid)
			return getkth(A, aStart + k / 2, B, bStart, k - k / 2);// Check:
																	// aRight +
																	// bLeft
		else
			return getkth(A, aStart, B, bStart + k / 2, k - k / 2);// Check:
																	// bRight +
																	// aLeft
	}

	// every number appears three times except one
	public int singleNumber(int[] nums) {
		int[] digits = new int[32];

		for (int i = 0; i < nums.length; i++) {
			int num = nums[i];
			for (int j = 31; j >= 0; j--) {
				digits[j] += num & 1;
				num >>= 1;
				if (num == 0)
					break;
			}
		}

		int res = 0;
		for (int i = 0; i < 32; i++) {
			if (digits[i] % 3 != 0)
				res |= 1 << (31 - i);
		}
		return res;
	}

	public int singleNumber2(int[] nums) {
		int res = 0;
		for (int i = 0; i < 32; i++) {
			int sum = 0;
			int x = 1 << i;
			for (int j = 0; j < nums.length; j++) {
				if ((nums[j] & x) != 0)
					sum++;
			}
			if (sum % 3 != 0)
				res |= x;
		}
		return res;
	}

	/*
	 * Given an array of numbers nums, in which exactly two elements appear only
	 * once and all the other elements appear exactly twice. Find the two
	 * elements that appear only once.
	 * 
	 * 
	 * sol:A^ B may look like this "0110...0010". The bit with "1" there is for
	 * sure coming from a "0" from A, and a "1" coming from B.
	 */

	public int[] singleNumberIII(int[] nums) {
		int xor = 0;
		for (int num : nums) {
			xor ^= num;
		}

		xor &= -xor;
		int[] res = { 0, 0 };
		for (int num : nums) {
			if ((num & xor) == 0)
				res[0] ^= num;
			else
				res[1] ^= num;
		}
		return res;
	}

	/*
	 * the bitwise and of the range is keeping the common bits of m and n from
	 * left to right until the first bit that they are different, padding zeros
	 * for the rest.
	 */

	public int rangeBitwiseAnd(int m, int n) {
		int i = 0;
		while (m != n) {
			m >>= 1;
			n >>= 1;
			i++;
		}
		return n << i;
	}

	public boolean isMatch(String s, String p) {
		boolean opt[][] = new boolean[s.length() + 1][p.length() + 1];
		// base case
		opt[0][0] = true;
		// first row
		for (int j = 2; j <= p.length(); j += 2) {
			if (p.charAt(j - 1) == '*')
				opt[0][j] = true;
			else
				break;
		}
		// iteration
		for (int i = 1; i <= s.length(); i++) {
			for (int j = 1; j <= p.length(); j++) {
				// below are all the cases for opt[i][j] to be true
				if (s.charAt(i - 1) == p.charAt(j - 1)
						|| p.charAt(j - 1) == '.')
					opt[i][j] = opt[i - 1][j - 1];
				else if (p.charAt(j - 1) == '*') {
					if (s.charAt(i - 1) == p.charAt(j - 2)
							|| p.charAt(j - 2) == '.')
						opt[i][j] = opt[i - 1][j - 2] /*
													 * if we absorb s[i-1] into
													 * p[j-2] & p[j-1]
													 */
								|| opt[i][j - 2] /*
												 * or if we take p[j-2] & p[j-1]
												 * as ""
												 */;
					else
						opt[i][j] = opt[i][j - 2] /*
												 * we can't use s[i-1] to match
												 * p[j-2] & p[j-1]
												 */;
				}
				// if none of above satisfies, opt[i][j] = false;
			}
		}
		return opt[s.length()][p.length()];
	}

	/*
	 * '.' Matches any single character. '*' Matches zero or more of the
	 * preceding element.
	 */

	public boolean isMatch2(String s, String p) {
		boolean dp[][] = new boolean[s.length() + 1][p.length() + 1];
		dp[0][0] = true;

		for (int i = 2; i <= p.length(); i += 2) {
			if (p.charAt(i - 1) == '*')
				dp[0][i] = true;
			else
				break;
		}

		for (int i = 1; i <= s.length(); i++) {
			for (int j = 1; j <= p.length(); j++) {
				char sc = s.charAt(i - 1);
				char pc = p.charAt(j - 1);
				if (sc == pc || pc == '.')
					dp[i][j] = dp[i - 1][j - 1];
				else if (pc == '*') {
					if (sc == p.charAt(j - 2) || p.charAt(j - 2) == '.')
						dp[i][j] = dp[i - 1][j] || dp[i][j - 2];
					else
						dp[i][j] = dp[i][j - 2];
				}
			}
		}
		return dp[s.length()][p.length()];
	}

	/*
	 * 1. Regular case: if two chars are the same, or p is ‘?’, then go to check
	 * dp[i-1][j-1]
	 * 
	 * 2. Special case: when p is ‘*’, we need to check dp[i][j-1] for every
	 * i>0. If there is one true, then the answer is true.
	 */

	public boolean isMatchWild(String s, String p) {
		int m = s.length(), n = p.length();
		boolean[][] dp = new boolean[m + 1][n + 1];
		dp[0][0] = true;
		for (int i = 1; i <= n; i++) {
			if (p.charAt(i - 1) == '*')
				dp[0][i] = true;
			else
				break;
		}
		for (int i = 1; i <= m; i++) {
			for (int j = 1; j <= n; j++) {
				if (s.charAt(i - 1) == p.charAt(j - 1)
						|| p.charAt(j - 1) == '?')
					dp[i][j] = dp[i - 1][j - 1];
				else if (p.charAt(j - 1) == '*') {
					int cur = i;
					while (cur >= 0) {
						if (dp[cur--][j - 1]) {
							dp[i][j] = true;
							break;
						}
					}
				}
			}
		}
		return dp[m][n];
	}

	public boolean isMatchWild2(String s, String p) {
		int m = s.length(), n = p.length();
		boolean[][] dp = new boolean[m + 1][n + 1];
		dp[0][0] = true;
		for (int i = 1; i <= n; i++) {
			if (p.charAt(i - 1) == '*')
				dp[0][i] = true;
			else
				break;
		}

		for (int i = 1; i <= m; i++) {
			for (int j = 1; j <= n; j++) {
				if (p.charAt(j - 1) != '*') {
					if (s.charAt(i - 1) == p.charAt(j - 1)
							|| p.charAt(j - 1) == '?')
						dp[i][j] = dp[i - 1][j - 1];
				} else {
					dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
				}
			}
		}
		return dp[m][n];
	}

	public int[] maxSlidingWindow(int[] nums, int k) {
		if (nums == null || k <= 0) {
			return new int[0];
		}
		int[] res = new int[nums.length - k + 1];
		PriorityQueue<Point> heap = new PriorityQueue<Point>(k,
				new Comparator<Point>() {
					@Override
					public int compare(Point p1, Point p2) {
						// TODO Auto-generated method stub
						return p2.x - p1.x;
					}
				});
		for (int i = 0; i < k; i++) {
			heap.offer(new Point(nums[i], i));
		}

		int j = 0;
		for (int i = k; i < nums.length; i++) {
			Point p = heap.peek();
			res[j++] = p.x;
			while (!heap.isEmpty() && p.y <= i - k) {
				heap.poll();
				p = heap.peek();
			}
			heap.offer(new Point(nums[i], i));
		}
		res[j] = heap.peek().x;
		return res;
	}

	public int[] maxSlidingWindow2(int[] nums, int k) {
		if (nums.length == 0 || k <= 0)
			return new int[0];
		int[] res = new int[nums.length - k + 1];
		Deque<Integer> que = new ArrayDeque<Integer>();
		int idx = 0;
		for (int i = 0; i < nums.length; i++) {
			while (!que.isEmpty() && que.peek() <= i - k)
				que.poll();
			while (!que.isEmpty() && nums[que.peekLast()] < nums[i])
				que.pollLast();
			que.offer(i);
			if (i >= k - 1)
				res[idx++] = nums[que.peek()];
		}
		return res;
	}

	public void recoverTree(TreeNode root) {
		if (root == null)
			return;
		TreeNode first = null, second = null;
		Stack<TreeNode> stk = new Stack<TreeNode>();
		TreeNode cur = root;
		while (cur != null) {
			stk.push(cur);
			cur = cur.left;
		}
		TreeNode pre = null;
		while (!stk.isEmpty()) {
			TreeNode top = stk.pop();
			if (pre != null) {
				if (pre.val >= top.val) {
					if (first == null)
						first = pre;
					second = top;
				}
			}
			pre = top;
			if (top.right != null) {
				top = top.right;
				while (top != null) {
					stk.push(top);
					top = top.left;
				}
			}
		}
		int t = first.val;
		first.val = second.val;
		second.val = t;
	}

	TreeNode pre = null, first = null, second = null;

	public void recoverTreeRecursive(TreeNode root) {
		recoverTreeUtil(root);
		int temp = first.val;
		first.val = second.val;
		second.val = temp;
	}

	public void recoverTreeUtil(TreeNode root) {
		if (root == null)
			return;
		recoverTreeUtil(root.left);
		if (pre != null && pre.val >= root.val) {
			if (first == null)
				first = pre;
			second = root;
		}
		pre = root;
		recoverTreeUtil(root.right);
	}

	public boolean validTree(int n, int[][] edges) {
		UnionFind uf = new UnionFind(n);
		for (int i = 0; i < edges.length; i++) {
			if (!uf.union(edges[i][0], edges[i][1]))
				return false;
		}
		System.out.println(Arrays.toString(uf.ids));
		return uf.count() == 1;
	}

	public boolean validTreeDFS(int n, int[][] edges) {
		List<List<Integer>> adjList = new ArrayList<List<Integer>>();
		for (int i = 0; i < n; i++) {
			adjList.add(new ArrayList<Integer>());
		}

		for (int[] edge : edges) {
			adjList.get(edge[0]).add(edge[1]);
			adjList.get(edge[1]).add(edge[0]);
		}

		boolean[] visited = new boolean[n];
		if (hasCycle(0, -1, visited, adjList))
			return false;
		for (int i = 0; i < n; i++) {
			if (!visited[i])
				return false;
		}
		return true;
	}

	public boolean hasCycle(int vertexId, int parent, boolean[] visited,
			List<List<Integer>> adjList) {
		if (visited[vertexId])
			return true;
		visited[vertexId] = true;
		List<Integer> neighbors = adjList.get(vertexId);
		for (int neighbor : neighbors) {
			if (neighbor != parent
					&& hasCycle(neighbor, vertexId, visited, adjList))
				return true;
		}
		return false;
	}

	public boolean validTreeBFS(int n, int[][] edges) {
		List<List<Integer>> adjList = new ArrayList<List<Integer>>();
		for (int i = 0; i < n; i++) {
			adjList.add(new ArrayList<Integer>());
		}

		for (int[] edge : edges) {
			adjList.get(edge[0]).add(edge[1]);
			adjList.get(edge[1]).add(edge[0]);
		}

		boolean[] visited = new boolean[n];
		Queue<Integer> que = new LinkedList<Integer>();
		que.offer(0);

		while (!que.isEmpty()) {
			int cur = que.poll();
			if (visited[cur])
				return false;
			visited[cur] = true;
			List<Integer> neighbors = adjList.get(cur);
			for (int neighbor : neighbors) {
				if (!visited[neighbor])
					que.offer(neighbor);
			}
		}
		for (int i = 0; i < n; i++) {
			if (!visited[i])
				return false;
		}
		return true;
	}

	public int nthUglyNumber(int n) {
		Queue<Integer> q1 = new LinkedList<Integer>();
		Queue<Integer> q2 = new LinkedList<Integer>();
		Queue<Integer> q3 = new LinkedList<Integer>();
		q1.offer(1);
		q2.offer(1);
		q3.offer(1);
		int m = 1;
		for (int i = 0; i < n; i++) {
			m = Math.min(Math.min(q1.peek(), q2.peek()), q3.peek());
			if (m == q1.peek())
				q1.poll();
			if (m == q2.peek())
				q2.poll();
			if (m == q3.peek())
				q3.poll();
			q1.offer(m * 2);
			q2.offer(m * 3);
			q3.offer(m * 5);
		}
		return m;
	}

	// O(n) dp
	public int nthUglyNumber2(int n) {
		int[] dp = new int[n - 1];
		dp[0] = 1;
		int f2 = 2, f3 = 3, f5 = 5;
		int ix2 = 0, ix3 = 0, ix5 = 0;

		for (int i = 1; i < n; i++) {
			int minV = Math.min(Math.min(f2, f3), f5);
			dp[i] = minV;

			if (minV == f2)
				f2 = 2 * dp[++ix2];
			if (minV == f3)
				f3 = 3 * dp[++ix3];
			if (minV == f5)
				f5 = 5 * dp[++ix5];
		}
		return dp[n - 1];

		// int[] dp=new int[n];
		// int f2=1, f3=1, f5=1;
		// int ix2=0,ix3=0,ix5=0;
		//
		// for(int i=0;i<n;i++){
		// int minV=Math.min(Math.min(f2, f3), f5);
		// dp[i]=minV;
		//
		// if(minV==f2)
		// f2=2*dp[ix2++];
		// if(minV==f3)
		// f3=3*dp[ix3++];
		// if(minV==f5)
		// f5=5*dp[ix5++];
		// }
		// return dp[n-1];
	}

	// idxs records the indices of each prime number in primes, similar to ugly
	// number 2
	public int nthSuperUglyNumber(int n, int[] primes) {
		int[] res = new int[n];
		int len = primes.length;
		int[] idxs = new int[len];
		res[0] = 1;
		for (int i = 1; i < n; i++) {
			int lastUglyNum = res[i - 1];
			for (int j = 0; j < len; j++) {
				while (res[idxs[j]] * primes[j] <= lastUglyNum)
					idxs[j]++;
			}
			int min = Integer.MAX_VALUE;
			for (int j = 0; j < len; j++) {
				if (res[idxs[j]] * primes[j] < min)
					min = res[idxs[j]] * primes[j];
			}
			res[i] = min;
		}
		return res[n - 1];
	}

	public ListNode oddEvenList(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode evenHead = new ListNode(0);
		ListNode oddHead = new ListNode(1);

		ListNode even = evenHead, odd = oddHead;

		int idx = 1;
		ListNode cur = head;
		while (cur != null) {
			if (idx % 2 == 0) {
				even.next = cur;
				even = even.next;
			} else {
				odd.next = cur;
				odd = odd.next;
			}
			cur = cur.next;
			idx++;
		}
		odd.next = evenHead.next;
		even.next = null;
		return oddHead.next;

	}

	public ListNode oddEvenList2(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode odd = head, evenHead = head.next;
		ListNode even = evenHead;
		while (even != null && even.next != null) {
			odd.next = even.next;
			even.next = even.next.next;
			odd = odd.next;
			even = even.next;
		}
		odd.next = evenHead;
		return head;
	}

	public boolean isScramble(String s1, String s2) {
		if (s1.length() != s2.length())
			return false;
		int len = s1.length();
		boolean[][][] dp = new boolean[len][len][len + 1];

		for (int i = len - 1; i >= 0; i--) {
			for (int j = len - 1; j >= 0; j--) {
				for (int k = 1; k <= len - Math.max(i, j); k++) {
					if (s1.substring(i, i + k).equals(s2.substring(j, j + k)))
						dp[i][j][k] = true;
					else {
						for (int l = 1; l < k; l++) {
							if (dp[i][j][l] && dp[i + l][j + l][k - l]
									|| dp[i][j + k - l][l]
									&& dp[i + l][j][k - l]) {
								dp[i][j][k] = true;
								break;
							}
						}
					}
				}
			}
		}
		return dp[0][0][len];
	}

	public int calculate2(String s) {
		Stack<Integer> stk = new Stack<Integer>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (Character.isDigit(c)) {
				int sum = c - '0';
				while (i + 1 < s.length() && Character.isDigit(s.charAt(i + 1))) {
					sum = sum * 10 + (s.charAt(i + 1) - '0');
					i++;
				}
				if (!stk.isEmpty() && (stk.peek() == 2 || stk.peek() == 3)) {
					int op = stk.pop();
					int firstNum = stk.pop();
					if (op == 2)
						stk.push(firstNum * sum);
					else if (op == 3)
						stk.push(firstNum / sum);
				} else
					stk.push(sum);
			} else {
				if (c == '+')
					stk.push(0);
				else if (c == '-')
					stk.push(1);
				else if (c == '*')
					stk.push(2);
				else if (c == '/')
					stk.push(3);
			}
		}
		int res = 0;
		while (!stk.isEmpty()) {
			if (stk.size() > 1) {
				int num = stk.pop();
				int op = stk.pop();
				if (op == 0)
					res += num;
				else if (op == 1)
					res -= num;
			} else
				res += stk.pop();
		}
		return res;
	}

	public List<String> fullJustify(String[] words, int maxWidth) {
		List<String> res = new ArrayList<String>();
		int n = words.length, i = 0;
		while (i < n) {
			int lineLen = 0;
			int newLen = 0;
			int beg = i;
			while (i < n) {
				if (lineLen == 0)
					newLen = words[i].length();
				else
					newLen = lineLen + 1 + words[i].length();
				if (newLen <= maxWidth)
					lineLen = newLen;
				else
					break;
				i++;
			}

			int spaces = maxWidth - lineLen;
			int spaceforeach = 0;
			if (i - beg - 1 > 0 && i < n) {
				spaceforeach = spaces / (i - beg - 1);
				spaces %= i - beg - 1;
			}
			String line = words[beg];
			for (int j = beg + 1; j < i; j++) {
				for (int k = 0; k <= spaceforeach; k++)
					line += " ";
				if (spaces > 0 && i < n) {
					line += " ";
					spaces--;
				}
				line += words[j];
			}
			// 下面这个for循环作用于一行只有一个单词还没填满一行的情况
			for (int j = 0; j < spaces; j++) {
				line += " ";
			}
			res.add(line);
		}
		return res;
	}

	public List<String> fullJustify2(String[] words, int maxWidth) {
		List<String> res = new ArrayList<String>();
		int i = 0, n = words.length;
		while (i < n) {
			int sum = 0, start = i;
			while (i < n && sum + words[i].length() <= maxWidth) {
				sum += words[i].length() + 1;
				i++;
			}
			int end = i - 1;
			int intervals = end - start;
			int avgSp = 0, leftSp = 0;
			if (intervals > 0) {
				avgSp = (maxWidth - sum + intervals + 1) / intervals;
				leftSp = (maxWidth - sum + intervals + 1) % intervals;
			}

			String line = words[start];
			for (int j = start + 1; j <= end; j++) {
				if (i == words.length) // last line
					line += " ";
				else {
					for (int k = 0; k < avgSp; k++)
						line += " ";
					if (leftSp > 0) {
						line += " ";
						leftSp--;
					}
				}
				line += words[j];
			}
			int left = maxWidth - line.length();
			if (left > 0) {
				for (int j = 0; j < left; j++) {
					line += " ";
				}
			}
			res.add(line);
		}
		return res;
	}

	public boolean isNum(char c) {
		return c >= '0' && c <= '9';
	}

	public boolean isSign(char c) {
		return c == '+' || c == '-';
	}

	public boolean isE(char c) {
		return c == 'e' || c == 'E';
	}

	public boolean isDot(char c) {
		return c == '.';
	}

	public boolean isNumber(String s) {
		s = s.trim();
		int pos = 0;
		boolean haveNum = false;
		// check sign
		if (pos < s.length() && isSign(s.charAt(pos)))
			pos++;
		// check number before a dot '.'
		while (pos < s.length() && isNum(s.charAt(pos))) {
			haveNum = true;
			pos++;
		}
		// check the dot
		if (pos < s.length() && isDot(s.charAt(pos))) {
			pos++;
		}
		// check numbers after a dot '.'
		while (pos < s.length() && isNum(s.charAt(pos))) {
			haveNum = true;
			pos++;
		}
		// check e/E
		if (haveNum && pos < s.length() && isE(s.charAt(pos))) {
			haveNum = false;
			pos++;
			if (pos < s.length() && isSign(s.charAt(pos))) {
				pos++;
			}
		}
		// check the number after a 'e'/'E'
		while (pos < s.length() && isNum(s.charAt(pos))) {
			haveNum = true;
			pos++;
		}
		// Everything is done, if not reach the end of string, return false.
		return pos == s.length() && haveNum;
	}

	public int[] findOrder(int numCourses, int[][] prerequisites) {
		int[] order = new int[numCourses];
		int[] indegrees = new int[numCourses];
		List<List<Integer>> graph = new ArrayList<List<Integer>>();
		Queue<Integer> que = new LinkedList<Integer>();
		for (int i = 0; i < numCourses; i++) {
			graph.add(new ArrayList<Integer>());
		}

		for (int i = 0; i < prerequisites.length; i++) {
			int course = prerequisites[i][0];
			int pre = prerequisites[i][1];
			indegrees[course]++;
			graph.get(pre).add(course);
		}

		for (int i = 0; i < indegrees.length; i++) {
			if (indegrees[i] == 0)
				que.offer(i);
		}
		int count = 0;
		while (!que.isEmpty()) {
			int pre = que.poll();
			order[count++] = pre;
			for (int course : graph.get(pre)) {
				if (--indegrees[course] == 0)
					que.offer(course);
			}
		}
		if (count == numCourses)
			return order;
		return new int[0];
	}

	public int[] findOrderDFS(int numCourses, int[][] prerequisites) {
		int[] visited = new int[numCourses];
		Stack<Integer> stk = new Stack<Integer>();
		List<List<Integer>> graph = new ArrayList<List<Integer>>();
		for (int i = 0; i < numCourses; i++) {
			graph.add(new ArrayList<Integer>());
		}
		for (int i = 0; i < prerequisites.length; i++) {
			int course = prerequisites[i][0];
			int pre = prerequisites[i][1];
			graph.get(pre).add(course);
		}

		for (int i = 0; i < numCourses; i++) {
			if (visited[i] == 0 && detectCycle(i, visited, stk, graph))
				return new int[0];
		}
		int[] res = new int[numCourses];
		for (int i = 0; i < numCourses; i++) {
			res[i] = stk.pop();
		}
		return res;
	}

	public boolean detectCycle(int prereq, int[] visited, Stack<Integer> stk,
			List<List<Integer>> graph) {
		visited[prereq] = -1;
		for (int course : graph.get(prereq)) {
			if (visited[course] == -1)
				return true;
			if (visited[course] == 0
					&& detectCycle(course, visited, stk, graph))
				return true;
		}
		visited[prereq] = 1;
		stk.push(prereq);
		return false;
	}

	public boolean canFinish3(int numCourses, int[][] prerequisites) {
		int count = 0;
		int[] indegrees = new int[numCourses];
		List<List<Integer>> graph = new ArrayList<List<Integer>>();
		Queue<Integer> que = new LinkedList<Integer>();
		for (int i = 0; i < numCourses; i++) {
			graph.add(new ArrayList<Integer>());
		}

		for (int i = 0; i < prerequisites.length; i++) {
			int course = prerequisites[i][0], pre = prerequisites[i][1];
			indegrees[course]++;
			graph.get(pre).add(course);
		}

		for (int i = 0; i < indegrees.length; i++) {
			if (indegrees[i] == 0)
				que.add(i);
		}

		while (!que.isEmpty()) {
			int pre = que.poll();
			count++;
			for (int course : graph.get(pre)) {
				if (--indegrees[course] == 0)
					que.offer(course);
			}
		}
		return count == numCourses;
	}

	public boolean canFinishDFS(int numCourses, int[][] prerequisites) {
		List<List<Integer>> graph = new ArrayList<List<Integer>>();
		int[] visited = new int[numCourses];
		for (int i = 0; i < numCourses; i++) {
			graph.add(new ArrayList<Integer>());
		}
		for (int i = 0; i < prerequisites.length; i++) {
			int course = prerequisites[i][0], pre = prerequisites[i][1];
			graph.get(pre).add(course);
		}
		for (int i = 0; i < numCourses; i++) {
			if (visited[i] == 0 && dfsDetectCycle(visited, graph, i))// white: 0
				return false;
		}
		return true;
	}

	public boolean dfsDetectCycle(int[] visited, List<List<Integer>> graph,
			int pre) {
		visited[pre] = -1;// gray:-1
		for (int course : graph.get(pre)) {
			if (visited[course] == -1)
				return true;
			if (visited[course] == 0 && dfsDetectCycle(visited, graph, course))
				return true;
		}
		visited[pre] = 1;// black:1
		return false;
	}

	public String removeDuplicateLetters(String s) {
		int[] count = new int[256];
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			count[c]++;
		}
		boolean[] visited = new boolean[256];
		Stack<Character> stk = new Stack<Character>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			count[c]--;
			if (visited[c])
				continue;

			while (!stk.isEmpty() && c < stk.peek() && count[stk.peek()] != 0) {
				visited[stk.peek()] = false;
				stk.pop();
			}
			stk.push(c);
			visited[c] = true;
		}
		StringBuilder sb = new StringBuilder();
		while (!stk.isEmpty()) {
			sb.insert(0, stk.pop());
		}
		return sb.toString();
	}

	// public int longestIncreasingPath(int[][] matrix) {
	// int m=matrix.length;
	// if(m==0)
	// return 0;
	// int n=matrix[0].length;
	//
	// int max=0;
	// for(int i=0;i<m;i++){
	// for(int j=0;j<n;j++){
	// boolean[][] visited=new boolean[m][n];
	// max=Math.max(max, longestIncreasingPathUtil(matrix, i, j, visited, 0));
	// }
	// }
	// return max;
	// }
	//
	// public int longestIncreasingPathUtil(int[][] matrix, int i, int j,
	// boolean[][]visited, int max){
	// int m=max;
	// visited[i][j]=true;
	// if(i>0&&matrix[i-1][j]>matrix[i][j]){
	// max++;
	// m=Math.max(m,longestIncreasingPathUtil(matrix, i-1, j, visited, max));
	// }
	// if(i<matrix.length-1&&matrix[i+1][j]>matrix[i][j]){
	// max++;
	// m=Math.max(m,longestIncreasingPathUtil(matrix, i+1, j, visited, max));
	// }
	//
	// if(j>0&&matrix[i][j-1]>matrix[i][j]){
	// max++;
	// m=Math.max(m,longestIncreasingPathUtil(matrix, i, j-1, visited, max));
	// }
	//
	// if(j<matrix[0].length-1&&matrix[i][j+1]>matrix[i][j]){
	// max++;
	// m=Math.max(m,longestIncreasingPathUtil(matrix, i, j+1, visited, max));
	// }
	// return m;
	// }

	int[] dx = { 1, -1, 0, 0 };
	int[] dy = { 0, 0, 1, -1 };
	int[][] dp;

	public int longestIncreasingPath(int[][] matrix) {
		int m = matrix.length;
		if (m == 0)
			return 0;
		int n = matrix[0].length;
		dp = new int[m][n];
		int max = 0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				max = Math.max(max, longestIncreasingPathUtil(matrix, i, j));
			}
		}
		return max;
	}

	public int longestIncreasingPathUtil(int[][] matrix, int i, int j) {
		if (dp[i][j] > 0)
			return dp[i][j];
		dp[i][j] = 1;
		for (int k = 0; k < 4; k++) {
			int x = i + dx[k], y = j + dy[k];
			if (x >= 0 && x < matrix.length && y >= 0 && y < matrix[0].length
					&& matrix[x][y] > matrix[i][j]) {
				dp[i][j] = Math.max(dp[i][j],
						1 + longestIncreasingPathUtil(matrix, x, y));
			}
		}
		return dp[i][j];
	}

	public int calculateMinimumHP(int[][] dungeon) {
		int m = dungeon.length;
		if (m == 0)
			return 0;
		int n = dungeon[0].length;
		int[][] dp = new int[m][n];
		dp[m - 1][n - 1] = Math.max(0 - dungeon[m - 1][n - 1], 0);
		for (int i = m - 2; i >= 0; i--) {
			dp[i][n - 1] = Math.max(dp[i + 1][n - 1] - dungeon[i][n - 1], 0);
		}

		for (int i = n - 2; i >= 0; i--) {
			dp[m - 1][i] = Math.max(dp[m - 1][i + 1] - dungeon[m - 1][i], 0);
		}

		for (int i = m - 2; i >= 0; i--) {
			for (int j = n - 2; j >= 0; j--) {
				dp[i][j] = Math.max(Math.min(dp[i + 1][j], dp[i][j + 1])
						- dungeon[i][j], 0);
			}
		}
		return dp[0][0] + 1;
	}

	public int minArea(char[][] image, int x, int y) {
		int m = image.length;
		if (m == 0)
			return 0;
		int[] res = new int[4];
		res[0] = m - 1;
		res[1] = 0;
		res[2] = image[0].length - 1;
		res[3] = 0;
		dfsMinArea(image, x, y, res);
		return (res[1] - res[0] + 1) * (res[3] - res[2] + 1);
	}

	public void dfsMinArea(char[][] image, int x, int y, int[] res) {
		if (x < 0 || x >= image.length || y < 0 || y >= image[0].length
				|| image[x][y] == '0')
			return;
		image[x][y] = '0';
		if (x < res[0])
			res[0] = x;
		if (x > res[1])
			res[1] = x;
		if (y < res[2])
			res[2] = y;
		if (y > res[3])
			res[3] = y;
		dfsMinArea(image, x + 1, y, res);
		dfsMinArea(image, x - 1, y, res);
		dfsMinArea(image, x, y + 1, res);
		dfsMinArea(image, x, y - 1, res);
	}

	public int minArea2(char[][] image, int x, int y) {
		int m = image.length;
		if (m == 0)
			return 0;
		int n = image[0].length;

		int minCol = searchColumns(image, 0, y, 0, m, true);
		int maxCol = searchColumns(image, y + 1, n, 9, m, false);
		int minRow = searchRows(image, 0, x, 0, n, true);
		int maxRow = searchRows(image, x + 1, m, 0, n, false);
		return (maxCol - minCol) * (maxRow - minRow);
	}

	public int searchColumns(char[][] image, int i, int j, int top, int bottom,
			boolean opt) {
		while (i != j) {
			int mid = (i + j) / 2, k = top;
			while (k < bottom && image[k][mid] == '0')
				k++;
			if (k < bottom == opt)
				j = mid;
			else
				i = mid + 1;
		}
		return i;
	}

	public int searchRows(char[][] image, int i, int j, int left, int right,
			boolean opt) {
		while (i != j) {
			int mid = (i + j) / 2, k = left;
			while (k < right && image[mid][k] == '0') {
				k++;
			}
			if (k < right == opt)
				j = mid;
			else
				i = mid + 1;
		}
		return i;
	}

	// google interview
	// 第一题若干人赛跑，已知是一个pair list, 譬如（1，4）表示1在4之前到达，求最后的排名
	public List<Integer> raceRanking(List<Point> pairs) {
		int n = pairs.size();
		Set<Integer> num = new HashSet<Integer>();
		for (int i = 0; i < n; i++) {
			int a = pairs.get(i).x;
			int b = pairs.get(i).y;
			if (!num.contains(a))
				num.add(a);
			if (!num.contains(b))
				num.add(b);
		}
		List<List<Integer>> graph = new ArrayList<List<Integer>>();
		for (int i = 0; i < num.size(); i++) {
			graph.add(new ArrayList<Integer>());
		}
		int[] count = new int[num.size()];
		for (int i = 0; i < pairs.size(); i++) {
			int src = pairs.get(i).x;
			int dst = pairs.get(i).y;
			count[pairs.get(i).y]++;
			graph.get(src).add(dst);
		}

		Queue<Integer> que = new LinkedList<Integer>();
		for (int i = 0; i < count.length; i++) {
			if (count[i] == 0)
				que.add(i);
		}
		System.out.println(graph);
		List<Integer> order = new ArrayList<Integer>();
		while (!que.isEmpty()) {
			int first = que.poll();
			order.add(first);
			for (int second : graph.get(first)) {
				if (--count[second] == 0)
					que.add(second);
			}
		}
		return order;
	}

	public int minTotalDistance3(int[][] grid) {
		int m = grid.length, n = grid[0].length;
		int[] rowSum = new int[n], colSum = new int[m];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				rowSum[j] += grid[i][j];
				colSum[i] += grid[i][j];
			}
		}
		return minDistance1D(rowSum) + minDistance1D(colSum);
	}

	public int minDistance1D(int[] sum) {
		int i = 0, j = sum.length;
		int d = 0, left = 0, right = 0;

		while (i != j) {
			if (left < right) {
				d += left;
				left += sum[++i];
			} else {
				d += right;
				right += sum[--j];
			}
		}
		return d;
	}

	public List<Integer> findMinHeightTrees(int n, int[][] edges) {
		if (n == 1)
			return Collections.singletonList(0);
		List<Set<Integer>> graph = new ArrayList<Set<Integer>>();
		for (int i = 0; i < n; i++)
			graph.add(new HashSet<Integer>());
		for (int i = 0; i < edges.length; i++) {
			graph.get(edges[i][0]).add(edges[i][1]);
			graph.get(edges[i][1]).add(edges[i][0]);
		}

		List<Integer> leaves = new ArrayList<Integer>();
		for (int i = 0; i < n; i++) {
			if (graph.get(i).size() == 1)
				leaves.add(i);
		}

		while (n > 2) {
			n -= leaves.size();
			List<Integer> newLeaves = new ArrayList<Integer>();
			for (int i : leaves) {
				int j = graph.get(i).iterator().next();
				graph.get(j).remove(i);
				if (graph.get(j).size() == 1)
					newLeaves.add(j);
			}
			leaves = newLeaves;
		}
		return leaves;
	}

	public String alienOrder(String[] words) {
		int n = words.length;
		List<List<Integer>> graph = new ArrayList<List<Integer>>();
		int[] indegrees = new int[26];
		Arrays.fill(indegrees, -1);
		for (int i = 0; i < 26; i++) {
			graph.add(new ArrayList<Integer>());
		}
		for (int i = 0; i < n; i++) {
			for (char c : words[i].toCharArray()) {
				if (indegrees[c - 'a'] < 0)
					indegrees[c - 'a'] = 0;
			}
			if (i > 0) {
				String word1 = words[i - 1];
				String word2 = words[i];
				int len = Math.min(word1.length(), word2.length());
				for (int j = 0; j < len; j++) {
					int c1 = word1.charAt(j) - 'a';
					int c2 = word2.charAt(j) - 'a';
					System.out.println(c1 == c2);
					if (c1 != c2) {
						if (!graph.get(c1).contains(c2)) {
							graph.get(c1).add(c2);
							indegrees[c2]++;
							break;
						}
					}
				}
			}
		}

		Queue<Integer> que = new LinkedList<Integer>();
		for (int i = 0; i < 26; i++) {
			if (indegrees[i] == 0)
				que.add(i);
		}
		System.out.println(Arrays.toString(indegrees));
		StringBuilder sb = new StringBuilder();
		while (!que.isEmpty()) {
			int c = que.poll();
			sb.append((char) (c + 'a'));
			for (int t : graph.get(c)) {
				if (--indegrees[t] == 0)
					que.offer(t);
			}
		}
		for (int d : indegrees) {
			if (d > 0)
				return "";
		}
		return sb.toString();
	}

	// alien dictionary dfs

	/*
	 * visited[i] = -1. Not even exist. visited[i] = 0. Exist. Non-visited.
	 * visited[i] = 1. Visiting. visited[i] = 2. Visited.
	 */
	public String alienOrderDFS(String[] words) {
		boolean[][] adj = new boolean[26][26];
		int[] visited = new int[26];
		buildGraph(words, adj, visited);

		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < 26; i++) {
			if (visited[i] == 0) {
				if (!dfs(adj, visited, sb, i))
					return "";
			}
		}
		return sb.toString();
	}

	public boolean dfs(boolean[][] adj, int[] visited, StringBuilder sb, int i) {
		visited[i] = 1;
		for (int j = 0; j < 26; j++) {
			if (adj[i][j]) {
				if (visited[j] == 1)
					return false;
				if (visited[j] == 0)
					if (!dfs(adj, visited, sb, j))
						return false;
			}
		}
		visited[i] = 2;
		sb.append((char) (i + 'a'));
		// System.out.println("temp is "+(char)(i+'a')+" and sb is "+sb.toString());
		return true;
	}

	public void buildGraph(String[] words, boolean[][] adj, int[] visited) {
		Arrays.fill(visited, -1);
		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			for (char c : word.toCharArray())
				visited[c - 'a'] = 0;
			if (i > 0) {
				String word2 = words[i - 1];
				int len = Math.min(word.length(), word2.length());
				for (int j = 0; j < len; j++) {
					char c1 = word.charAt(j), c2 = word2.charAt(j);
					if (c1 != c2) {
						adj[c1 - 'a'][c2 - 'a'] = true;
						break;
					}
				}
			}
		}
	}

	public String topologicalSort(String[] words) {
		LinkedList<Integer>[] adj = new LinkedList[26];
		buildGraph(words, adj);

		Stack<Integer> stk = new Stack<Integer>();
		boolean[] visited = new boolean[26];

		for (int i = 0; i < 26; i++) {
			if (!visited[i] && adj[i] != null) {
				toplogicalSortUtil(adj, visited, stk, i);
			}
		}
		StringBuilder sb = new StringBuilder();
		while (!stk.isEmpty()) {
			sb.append((char) (stk.pop() + 'a'));
		}
		return sb.toString();
	}

	public void toplogicalSortUtil(LinkedList<Integer>[] adj,
			boolean[] visited, Stack<Integer> stk, int i) {
		visited[i] = true;
		Iterator<Integer> it = adj[i].iterator();
		while (it.hasNext()) {
			int v = it.next();
			if (!visited[v]) {
				toplogicalSortUtil(adj, visited, stk, v);
			}
		}
		stk.push(i);
		// System.out.println((char)(i+'a'));
	}

	public void buildGraph(String[] words, LinkedList<Integer>[] adj) {
		for (int i = 0; i < words.length; i++) {
			for (char c : words[i].toCharArray()) {
				adj[c - 'a'] = new LinkedList<Integer>();
			}
			if (i > 0) {
				String w1 = words[i - 1], w2 = words[i];
				int len = Math.min(w1.length(), w2.length());
				for (int j = 0; j < len; j++) {
					char c1 = w1.charAt(j), c2 = w2.charAt(j);
					if (c1 != c2) {
						adj[c1 - 'a'].add(c2 - 'a');
						break;
					}
				}
			}
		}
	}

	// make coins.
	public int count(int n, int[] coins) {
		int m = coins.length;
		int[][] dp = new int[m][n + 1];

		for (int i = 0; i < m; i++) {
			dp[i][0] = 1;
		}

		for (int i = 1; i <= n; i++) {
			for (int j = 0; j < m; j++) {
				int x = i >= coins[j] ? dp[j][i - coins[j]] : 0;
				int y = j > 0 ? dp[j - 1][i] : 0;
				dp[j][i] = x + y;
			}
		}
		return dp[m - 1][n];
	}

	/*
	 * 举个例子54215，比如现在求百位上的1，54215的百位上是2。可以看到xx100到xx199的百位上都是1，这里xx从0到54，即100->199
	 * , 1100->1199...54100->54199, 这些数的百位都是1，因此百位上的1总数是55*100
	 * 
	 * 如果n是54125,这时由于它的百位是1，先看xx100到xx199，其中xx是0到53，即54*100,
	 * 然后看54100到54125，这是26个。所以百位上的1的总数是54*100 + 26.
	 * 
	 * 如果n是54025，那么只需要看xx100到xx199中百位上的1，这里xx从0到53，总数为54*100
	 * 
	 * 求其他位的1的个数的方法是一样的。
	 */
	public int countDigitOne(int n) {
		if (n <= 0)
			return 0;
		int res = 0;
		long base = 1;
		while (n >= base) {
			long left = n / base;
			long right = n % base;
			if ((left % 10) > 1)
				res += (left / 10 + 1) * base;
			else if ((left % 10) == 1)
				res += (left / 10) * base + (right + 1);
			else
				res += (left / 10) * base;
			base *= 10;
		}
		return res;
	}

	public void wiggleSortII(int[] nums) {
		Arrays.sort(nums);
		int n = nums.length;
		int[] temp = Arrays.copyOf(nums, n);
		int mid = n % 2 == 0 ? n / 2 - 1 : n / 2;
		int index = 0;
		for (int i = 0; i <= mid; i++) {
			nums[index] = temp[mid - i];
			if (index + 1 < n) {
				nums[index + 1] = temp[n - i - 1];
			}
			index += 2;
		}
	}

	public boolean isPowerOfThree(int n) {
		double res = Math.log(n) / Math.log(3);
		return Math.abs(res - Math.rint(res)) < 0.000000000001;
	}

	public String getPermutation(int n, int k) {
		List<Integer> nums = new ArrayList<Integer>();
		int factorial = 1;
		for (int i = 1; i <= n; i++) {
			nums.add(i);
			factorial *= i;
		}
		k--;
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < n; i++) {
			factorial /= (n - i);
			int index = k / factorial;
			sb.append(nums.get(index));
			k %= factorial;
			nums.remove(index);
		}
		return sb.toString();
	}

	// count nodes in complete tree
	public int countNodesLC(TreeNode root) {
		if (root == null)
			return 0;
		int left = getLeftHeight(root.left);
		int right = getRightHeight(root.right);
		if (left == right)
			return (2 << left) - 1;
		return countNodesLC(root.left) + countNodesLC(root.right) + 1;
	}

	public int getLeftHeight(TreeNode root) {
		if (root == null)
			return 0;
		int h = 0;
		while (root != null) {
			h++;
			root = root.left;
		}
		return h;
	}

	public int getRightHeight(TreeNode root) {
		if (root == null)
			return 0;
		int h = 0;
		while (root != null) {
			h++;
			root = root.right;
		}
		return h;
	}

	// google interview
	/*
	 * * You are given two array, first array contain integer which represent
	 * heights of persons and second array contain how many persons in front of
	 * him are standing who are greater than him in term of height and forming a
	 * queue. Ex A: 3 2 1 B: 0 1 1 It means in front of person of height 3 there
	 * is no person standing, person of height 2 there is one person in front of
	 * him who has greater height then he, similar to person of height 1. Your
	 * task to arrange them Ouput should be. 3 1 2 Here - 3 is at front, 1 has 3
	 * in front ,2 has 1 and 3 in front.
	 */
	/**
	 * Main idea of this problem is that, when sorting in descending order, and
	 * we insert from the back to the front, wont affects the already existed
	 * order.
	 */

	public void arrangeHeight(int[] heights, int[] counts) {
		int n = heights.length;
		if (n < 2)
			return;
		List<HeightNode> nodes = new ArrayList<HeightNode>();
		for (int i = 0; i < n; i++) {
			nodes.add(new HeightNode(heights[i], counts[i]));
		}

		Collections.sort(nodes, new Comparator<HeightNode>() {
			@Override
			public int compare(HeightNode n1, HeightNode n2) {
				return n2.height - n1.height;
			}
		});

		for (int i = 1; i < n; i++) {
			HeightNode node = nodes.get(i);
			int index = node.tallerPeopleInFront;
			nodes.remove(i);
			nodes.add(index, node);
		}

		for (int i = 0; i < n; i++) {
			heights[i] = nodes.get(i).height;
		}
		System.out.println(Arrays.toString(heights));
	}

	public List<List<String>> findLadders(String beginWord, String endWord,
			Set<String> wordList) {
		List<List<String>> res = new ArrayList<List<String>>();
		Queue<String> curLevel = new LinkedList<String>();
		HashMap<String, List<String>> map = new HashMap<String, List<String>>();
		Set<String> visited = new HashSet<String>();
		boolean found = false;

		curLevel.add(beginWord);
		wordList.add(endWord);
		visited.add(beginWord);

		while (!curLevel.isEmpty()) {
			Set<String> toBuild = new HashSet<String>();
			Queue<String> nextLevel = new LinkedList<String>();
			while (!curLevel.isEmpty()) {
				String s = curLevel.poll();
				List<String> neighbors = new ArrayList<String>();
				char[] word = s.toCharArray();
				for (int i = 0; i < word.length; i++) {
					char t = word[i];
					for (char c = 'a'; c <= 'z'; c++) {
						if (word[i] != c) {
							word[i] = c;
							String st = new String(word);
							if (wordList.contains(st) && !visited.contains(st)) {
								neighbors.add(st);
								if (toBuild.add(st)) {
									nextLevel.add(st);
								}
							}
							found = found || st.equals(endWord);
						}
					}
					word[i] = t;
				}
				map.put(s, neighbors);
			}
			curLevel = nextLevel;
			visited.addAll(toBuild);
			if (found)
				break;
		}
		if (found) {
			dfsFindLadders(beginWord, endWord, map, new ArrayList<String>(),
					res);
		}
		return res;
	}

	public void dfsFindLadders(String beg, String end,
			HashMap<String, List<String>> map, List<String> sol,
			List<List<String>> res) {
		if (beg.equals(end)) {
			List<String> out = new ArrayList<String>(sol);
			out.add(end);
			res.add(out);
			return;
		}
		if (!map.containsKey(beg))
			return;
		sol.add(beg);
		for (String s : map.get(beg)) {
			dfsFindLadders(s, end, map, sol, res);
		}
		sol.remove(sol.size() - 1);
	}

	public List<List<String>> findLadders2(String beginWord, String endWord,
			Set<String> wordList) {
		List<List<String>> res = new ArrayList<List<String>>();
		Queue<String> que = new LinkedList<String>();
		int cur = 0, next = 0;

		Set<String> unvisited = new HashSet<String>(wordList);
		Set<String> visited = new HashSet<String>();
		boolean found = false;

		Map<String, List<String>> map = new HashMap<String, List<String>>();
		que.add(beginWord);
		cur++;
		unvisited.remove(beginWord);
		unvisited.add(endWord);

		while (!que.isEmpty()) {
			String s = que.remove();
			cur--;
			char[] word = s.toCharArray();
			for (int i = 0; i < word.length; i++) {
				char t = word[i];
				for (char c = 'a'; c <= 'z'; c++) {
					if (c != word[i]) {
						word[i] = c;
						String st = new String(word);
						if (unvisited.contains(st)) {
							if (visited.add(st)) {
								que.offer(st);
								next++;
							}
							if (map.containsKey(st))
								map.get(st).add(s);
							else {
								List<String> lst = new ArrayList<String>();
								lst.add(s);
								map.put(st, lst);
							}
							if (st.equals(endWord) && !found) {
								found = true;
							}
						}
					}
				}
				word[i] = t;
			}
			if (cur == 0) {
				if (found)
					break;
				cur = next;
				next = 0;
				unvisited.removeAll(visited);
				visited.clear();
			}
		}
		if (found)
			dfsLadder(endWord, beginWord, map, new ArrayList<String>(), res);
		return res;

	}

	public void dfsLadder(String word, String start,
			Map<String, List<String>> map, List<String> sol,
			List<List<String>> res) {
		if (word.equals(start)) {
			List<String> out = new ArrayList<String>(sol);
			out.add(0, start);
			res.add(out);
			return;
		}
		if (!map.containsKey(word))
			return;
		sol.add(0, word);
		for (String s : map.get(word)) {
			dfsLadder(s, start, map, sol, res);
		}
		sol.remove(0);
	}

	public List<List<Integer>> permutations(int n) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		boolean[] used = new boolean[n + 1];
		permutationsUtil(0, n, used, sol, res);
		return res;
	}

	public void permutationsUtil(int dep, int n, boolean[] used,
			List<Integer> sol, List<List<Integer>> res) {
		if (dep == n) {
			res.add(new ArrayList<Integer>(sol));
			return;
		}
		for (int i = 1; i <= n; i++) {
			if (!used[i]) {
				sol.add(i);
				used[i] = true;
				permutationsUtil(dep + 1, n, used, sol, res);
				sol.remove(sol.size() - 1);
				used[i] = false;
			}
		}
	}

	public List<Integer> possibleNums(List<Integer> nums) {
		List<Integer> res = new ArrayList<Integer>();
		if (nums.size() == 0)
			return res;
		possibleNumsUtil(1, nums, nums.get(0), res, 1);
		return res;
	}

	public void possibleNumsUtil(int dep, List<Integer> nums, int curval,
			List<Integer> res, int cur) {

		if (dep == nums.size()) {
			if (!res.contains(curval))
				res.add(curval);
			return;
		}
		for (int i = cur; i < nums.size(); i++) {
			possibleNumsUtil(dep + 1, nums, curval + nums.get(i), res, i + 1);
			possibleNumsUtil(dep + 1, nums, curval - nums.get(i), res, i + 1);
			possibleNumsUtil(dep + 1, nums, curval * nums.get(i), res, i + 1);
			possibleNumsUtil(dep + 1, nums, curval / nums.get(i), res, i + 1);
		}
	}

	/*
	 * paint fence 设S(n)表示当前杆和前一个杆颜色相同的个数，D(n)表示当前杆和前一个颜色不相同的个数。
	 * 
	 * 则递推关系式为：
	 * 
	 * S(n) = D(n - 1)， 即若当前杆和前一个杆颜色相同的个数等于前一个杆和再前一个杆颜色不相同的个数。
	 * 
	 * D(n) = (k - 1) * (D(n - 1) + S(n -
	 * 1))，即前一个杆和再前一个杆所有可能的颜色组合，乘以当前杆与前一个杆颜色不相同的个数，即（k - 1）。
	 */
	public int numWays(int n, int k) {
		if (n == 0 || k == 0)
			return 0;
		if (n == 1)
			return k;
		int lastS = k;
		int lastD = k * (k - 1);
		for (int i = 2; i < n; i++) {
			int temp = (lastS + lastD) * (k - 1);
			lastS = lastD;
			lastD = temp;
		}
		return lastS + lastD;
	}

	// paint house
	public int minCost(int[][] costs) {
		int n = costs.length;
		if (n == 0)
			return 0;
		for (int i = 1; i < n; i++) {
			costs[i][0] += Math.min(costs[i - 1][1], costs[i - 1][2]);
			costs[i][1] += Math.min(costs[i - 1][0], costs[i - 1][2]);
			costs[i][2] += Math.min(costs[i - 1][0], costs[i - 1][1]);
		}

		return Math.min(costs[n - 1][0],
				Math.min(costs[n - 1][1], costs[n - 1][2]));
	}

	public int minCost1(int[][] costs) {
		int n = costs.length;
		if (n == 0)
			return 0;
		int r = 0, g = 0, b = 0;
		for (int i = 0; i < n; i++) {
			int rr = r, bb = b, gg = g;
			r = costs[i][0] + Math.min(bb, gg);
			g = costs[i][1] + Math.min(rr, bb);
			b = costs[i][2] + Math.min(rr, gg);
		}

		return Math.min(r, Math.min(g, b));
	}

	// moving average google
	Queue<Integer> que = new LinkedList<Integer>();
	int queSum = 0;

	public int MovingAverage(int num, int n) {
		if (que.size() == n) {
			queSum -= que.poll();
		}
		que.offer(num);
		queSum += num;
		return queSum / que.size();
	}

	public int minCostII(int[][] costs) {
		int m = costs.length;
		if (m == 0)
			return 0;
		int n = costs[0].length;

		int preMin = 0, preSecMin = 0, preIdx = -1;

		for (int i = 0; i < m; i++) {
			int curMin = Integer.MAX_VALUE, curSecMin = Integer.MAX_VALUE, curIdx = -1;
			for (int j = 0; j < n; j++) {
				costs[i][j] += j == preIdx ? preSecMin : preMin;
				if (costs[i][j] < curMin) {
					curSecMin = curMin;
					curMin = costs[i][j];
					curIdx = j;
				} else if (costs[i][j] < curSecMin) {
					curSecMin = costs[i][j];
				}
			}
			preMin = curMin;
			preSecMin = curSecMin;
			preIdx = curIdx;
		}
		return preMin;
	}

	/*
	 * Google interview: A string consists of ‘0’, ‘1’ and '?'. The question
	 * mark can be either '0' or '1'. Find all possible combinations for a
	 * string.
	 */
	public List<String> findAllPossibleCombinations(String s) {
		List<String> res = new ArrayList<String>();
		int count = 0;
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == '?')
				count++;
		}
		findAllPossibleCombinationsUtil(0, count, s, "", res, 0);
		return res;
	}

	public void findAllPossibleCombinationsUtil(int dep, int maxDep, String s,
			String sol, List<String> res, int cur) {
		if (dep == maxDep) {
			res.add(sol + s.substring(cur));
			return;
		}
		for (int i = cur; i < s.length(); i++) {
			if (s.charAt(i) == '?') {
				findAllPossibleCombinationsUtil(dep + 1, maxDep, s, sol + '0',
						res, i + 1);
				findAllPossibleCombinationsUtil(dep + 1, maxDep, s, sol + '1',
						res, i + 1);
			} else {
				sol = sol + s.charAt(i);
			}
		}
	}

	public List<String> findAllPossibleCombinations2(String s) {
		List<String> res = new ArrayList<String>();
		int index = s.indexOf('?');
		if (index < 0) {
			res.add(s);
		} else {
			String s1 = s.substring(0, index) + '0' + s.substring(index + 1);
			String s2 = s.substring(0, index) + '1' + s.substring(index + 1);
			res.addAll(findAllPossibleCombinations2(s1));
			res.addAll(findAllPossibleCombinations2(s2));
		}
		return res;
	}

	public List<String> findAllPossibleCombinations3(String s) {
		List<String> res = new ArrayList<String>();
		findAllPossibleCombinations3Util("", s, 0, res);
		return res;
	}

	public void findAllPossibleCombinations3Util(String sol, String s, int cur,
			List<String> res) {
		if (sol.length() == s.length()) {
			res.add(sol);
		}

		for (int i = cur; i < s.length(); i++) {
			if (s.charAt(i) == '?') {
				findAllPossibleCombinations3Util(sol + "0", s, i + 1, res);
				findAllPossibleCombinations3Util(sol + "1", s, i + 1, res);
			} else {
				findAllPossibleCombinations3Util(sol + s.charAt(i), s, i + 1,
						res);
			}
		}
	}

	/*
	 * google interview
	 */
	// drop rain, get the expectation
	public int simulateRainDrop() {
		DropInterval[] intervals = new DropInterval[100];
		double start = 0.0, size = 0.01;
		for (int i = 0; i < 100; i++) {
			intervals[i] = new DropInterval(start, start + size);
			start += size;
		}

		int count = 0, wetCount = 0;

		while (wetCount < 100) {
			double center = Math.random();
			count++;
			double left = center - size / 2;
			double right = center + size / 2;

			if (left >= 0.0) {
				int index = (int) (left / 0.01);
				if (!intervals[index].isWet()) {
					if (left < intervals[index].right) {
						intervals[index].right = left;
						if (intervals[index].isWet())
							wetCount++;
					}
				}
			}
			if (right <= 1.0) {
				int index = (int) (right / 0.01);
				if (!intervals[index].isWet()) {
					if (right > intervals[index].left) {
						intervals[index].left = right;
						if (intervals[index].isWet())
							wetCount++;
					}
				}
			}
		}
		return count;
	}

	/*
	 * 给一个List<Stirng> l和一个number n，让
	 * rearrange这些string，使得每一个string之后至少n个位置，才能出现重复的string。 如果这种结果不止一个，返回任意一个
	 * 例如，List = {"a","a","b","b","c","c“},n = 3 return {a,b,c,a,b,c}
	 */
	class StringFreq {
		String s;
		int freq;

		public StringFreq(String s, int freq) {
			this.s = s;
			this.freq = freq;
		}
	}

	public List<String> rearrangeString(List<String> strs, int n) {
		Map<String, Integer> count = new HashMap<String, Integer>();
		for (String s : strs) {
			if (count.containsKey(s))
				count.put(s, count.get(s) + 1);
			else
				count.put(s, 1);
		}

		List<String> res = new ArrayList<String>();
		for (int i = 0; i < strs.size(); i++)
			res.add(null);
		PriorityQueue<StringFreq> heap = new PriorityQueue<StringFreq>(
				count.size(), new Comparator<StringFreq>() {

					@Override
					public int compare(StringFreq o1, StringFreq o2) {
						return o2.freq - o1.freq;
					}

				});
		for (String s : count.keySet()) {
			heap.offer(new StringFreq(s, count.get(s)));
		}

		int start = 0;
		for (int i = 0; i < count.size(); i++) {
			StringFreq sf = heap.poll();
			int index = start;
			while (sf.freq > 0) {
				if (index < strs.size() && res.get(index) == null)
					res.set(index, sf.s);
				else
					throw new IllegalArgumentException("invalid input!");
				index += n;
				sf.freq--;
			}
			start++;
		}
		return res;
	}

	/*
	 * http://blog.csdn.net/linhuanmars/article/details/23236995
	 * 
	 * local[i][j]和global[i][j]的区别是：local[i][j]意味着在第i天一定有交易（卖出）发生，当第i天的价格高于第i-1天（
	 * 即diff >
	 * 0）时，那么可以把这次交易（第i-1天买入第i天卖出）跟第i-1天的交易（卖出）合并为一次交易，即local[i][j]=local
	 * [i-1][j]
	 * +diff；当第i天的价格不高于第i-1天（即diff<=0）时，那么local[i][j]=global[i-1][j-1]+diff
	 * ，而由于diff
	 * <=0，所以可写成local[i][j]=global[i-1][j-1]。global[i][j]就是我们所求的前i天最多进行k次交易的最大收益
	 * ，
	 * 可分为两种情况：如果第i天没有交易（卖出），那么global[i][j]=global[i-1][j]；如果第i天有交易（卖出），那么global
	 * [i][j]=local[i][j]。
	 */
	public int maxProfitIV(int k, int[] prices) {
		int n = prices.length;
		if (k >= n / 2) {
			int res = 0;
			for (int i = 1; i < n; i++) {
				if (prices[i] > prices[i - 1]) {
					res += prices[i] - prices[i - 1];
				}
			}
			return res;
		}

		int[][] global = new int[n][k + 1];
		int[][] local = new int[n][k + 1];

		for (int i = 1; i < n; i++) {
			for (int j = 1; j <= k; j++) {
				local[i][j] = Math.max(
						global[i - 1][j - 1]
								+ Math.max(prices[i] - prices[i - 1], 0),
						local[i - 1][j] + prices[i] - prices[i - 1]);
				global[i][j] = Math.max(global[i - 1][j], local[i][j]);
			}
		}
		return global[n - 1][k];
	}

	// state machine thinking
	/*
	 * https://leetcode.com/discuss/72030/share-my-dp-solution-by-state-machine-
	 * thinking s0[i] = max(s0[i - 1], s2[i - 1]); // Stay at s0, or rest from
	 * s2 s1[i] = max(s1[i - 1], s0[i - 1] - prices[i]); // Stay at s1, or buy
	 * from s0 s2[i] = s1[i - 1] + prices[i]; // Only one way from s1
	 */
	public int maxProfitCoolDown(int[] prices) {
		int n = prices.length;
		if (n <= 1)
			return 0;
		int[] rest = new int[n];
		int[] buy = new int[n];
		int[] sell = new int[n];
		buy[0] = -prices[0];
		sell[0] = Integer.MIN_VALUE;

		for (int i = 1; i < n; i++) {
			rest[i] = Math.max(sell[i - 1], rest[i - 1]);
			buy[i] = Math.max(rest[i - 1] - prices[i], buy[i - 1]);
			sell[i] = buy[i - 1] + prices[i];
		}
		return Math.max(sell[n - 1], rest[n - 1]);
	}

	/*
	 * sellDp[i]=Math.max(sellDp[i-1], buyDp[i-1]+prices[i])
	 * buyDp[i]=Math.max(sellDp[i-2]-prices[i], buyDp[i-1])
	 */
	public int maxProfitCoolDownDP(int[] prices) {
		int n = prices.length;
		if (n <= 1)
			return 0;
		int[] sellDp = new int[n];
		int[] buyDp = new int[n];
		buyDp[0] = -prices[0];

		for (int i = 1; i < n; i++) {
			sellDp[i] = Math.max(sellDp[i - 1], buyDp[i - 1] + prices[i]);
			if (i >= 2)
				buyDp[i] = Math.max(buyDp[i - 1], sellDp[i - 2] - prices[i]);
			else
				buyDp[i] = Math.max(buyDp[i - 1], -prices[i]);
		}
		return sellDp[n - 1];
	}

	public int maxProfitCoolDownDP_O1(int[] prices) {
		if (prices == null || prices.length == 0) {
			return 0;
		}

		int currBuy = -prices[0];
		int currSell = 0;
		int prevSell = 0;
		for (int i = 1; i < prices.length; i++) {
			int temp = currSell;
			currSell = Math.max(currSell, currBuy + prices[i]);
			if (i >= 2) {
				currBuy = Math.max(currBuy, prevSell - prices[i]);
			} else {
				currBuy = Math.max(currBuy, -prices[i]);
			}
			prevSell = temp;
		}
		return currSell;
	}

	public List<int[]> getSkyline(int[][] buildings) {
		List<int[]> res = new ArrayList<int[]>();
		int n = buildings.length;
		if (n == 0)
			return res;
		List<int[]> heights = new ArrayList<int[]>();
		for (int[] b : buildings) {
			heights.add(new int[] { b[0], -b[2] });
			heights.add(new int[] { b[1], b[2] });
		}
		Collections.sort(heights, new Comparator<int[]>() {
			@Override
			public int compare(int[] h1, int[] h2) {
				if (h1[0] != h2[0])
					return h1[0] - h2[0];
				return h1[1] - h2[1];
			}
		});

		PriorityQueue<Integer> heap = new PriorityQueue<Integer>(n,
				Collections.reverseOrder());
		heap.offer(0);
		int pre = 0;
		for (int[] h : heights) {
			if (h[1] < 0)
				heap.offer(-h[1]);
			else
				heap.remove(h[1]);
			int cur = heap.peek();
			if (pre != cur) {
				res.add(new int[] { h[0], cur });
				pre = cur;
			}
		}
		return res;
	}

	public List<List<Integer>> getFactors(int n) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (n < 2)
			return res;
		List<Integer> sol = new ArrayList<Integer>();
		getFactorsUtil(n, sol, res, 2);
		return res;
	}

	public void getFactorsUtil(int n, List<Integer> sol,
			List<List<Integer>> res, int start) {
		if (n == 1) {
			if (sol.size() > 1) {
				res.add(new ArrayList<Integer>(sol));
			}
		}
		for (int i = start; i <= n; i++) {
			if (n % i == 0) {
				sol.add(i);
				getFactorsUtil(n / i, sol, res, i);
				sol.remove(sol.size() - 1);
			}
		}
	}

	public int shortestDistanceI(String[] words, String word1, String word2) {
		int idx1 = -1, idx2 = -1;
		int shortestDis = Integer.MAX_VALUE;
		for (int i = 0; i < words.length; i++) {
			if (words[i].equals(word1))
				idx1 = i;
			else if (words[i].equals(word2))
				idx2 = i;
			if (idx1 != -1 && idx2 != -1) {
				shortestDis = Math.min(shortestDis, Math.abs(idx1 - idx2));
			}
		}
		return shortestDis;
	}

	public int shortestWordDistanceIII(String[] words, String word1,
			String word2) {
		int minDis = Integer.MAX_VALUE, i1 = -1, i2 = -1;

		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			if (word.equals(word1))
				i1 = i;
			if (word.equals(word2)) {
				if (word1.equals(word2))
					i1 = i2;
				i2 = i;
			}
			System.out.println(i1 + ", " + i2);
			if (i1 != -1 && i2 != -1) {
				minDis = Math.min(minDis, Math.abs(i1 - i2));
			}
		}
		return (int) minDis;
	}

	public List<String> install(String[][] deps) {
		Map<String, Software> map = new HashMap<String, Software>();
		for (String[] dep : deps) {
			Software src = map.get(dep[0]);
			Software dst = map.get(dep[1]);
			if (src == null)
				src = new Software(dep[0]);
			if (dst == null)
				dst = new Software(dep[1]);
			src.targets.add(dst);
			dst.deps++;
			map.put(dep[0], src);
			map.put(dep[1], dst);
		}

		PriorityQueue<Software> que = new PriorityQueue<Software>(10,
				new Comparator<Software>() {
					@Override
					public int compare(Software s1, Software s2) {
						return s1.deps - s2.deps;
					}
				});

		for (String s : map.keySet()) {
			if (map.get(s).deps == 0)
				que.offer(map.get(s));
		}

		List<String> res = new ArrayList<String>();
		while (!que.isEmpty()) {
			Software s = que.poll();
			res.add(s.name);
			for (Software t : s.targets) {
				t.deps--;
				if (t.deps == 0)
					que.offer(t);
			}
		}
		return res;
	}

	class Software {
		public String name;
		public int deps;
		public List<Software> targets;

		public Software(String name) {
			this.name = name;
			deps = 0;
			targets = new ArrayList<Software>();
		}
	}

	public boolean isStrobogrammatic(String num) {
		if (num.length() == 0)
			return true;
		Map<Character, Character> map = new HashMap<Character, Character>();
		map.put('0', '0');
		map.put('1', '1');
		map.put('8', '8');
		map.put('6', '9');
		map.put('9', '6');
		int beg = 0, end = num.length() - 1;
		while (beg <= end) {
			char c1 = num.charAt(beg);
			char c2 = num.charAt(end);
			if (!map.containsKey(c1) || map.get(c1) != c2)
				return false;
			beg++;
			end--;
		}
		return true;
	}

	public List<String> findStrobogrammatic(int n) {
		return findStrobogrammaticUtil(n, n);
	}

	public List<String> findStrobogrammaticUtil(int n, int m) {
		if (n == 0)
			return new ArrayList<String>(Arrays.asList(""));
		if (n == 1)
			return new ArrayList<String>(Arrays.asList("0", "1", "8"));
		List<String> res = new ArrayList<String>();
		List<String> lst = findStrobogrammaticUtil(n - 2, m);

		for (String s : lst) {
			if (n != m)
				res.add("0" + s + "0");// 0 cannot be the first digit
			res.add("1" + s + "1");
			res.add("6" + s + "9");
			res.add("8" + s + "8");
			res.add("9" + s + "6");
		}
		return res;
	}

	public int strobogrammaticInRange(String low, String high) {
		int count = 0;
		List<String> res = new ArrayList<String>();
		for (int i = low.length(); i <= high.length(); i++) {
			res.addAll(findStrobogrammaticUtil(i, i));
		}

		for (String num : res) {
			if (num.length() == low.length() && num.compareTo(low) < 0
					|| num.length() == high.length() && num.compareTo(high) > 0)
				continue;
			count++;
		}
		return count;
	}

	int count = 0;
	char[][] pairs = { { '0', '0' }, { '1', '1' }, { '6', '9' }, { '8', '8' },
			{ '9', '6' } };

	public int strobogrammaticInRange2(String low, String high) {
		for (int len = low.length(); len <= high.length(); len++) {
			strobogrammaticUtil(low, high, new char[len], 0, len - 1);
		}
		return count;
	}

	public void strobogrammaticUtil(String low, String high, char[] ch,
			int left, int right) {
		if (left > right) {
			String s = new String(ch);
			if (s.length() == low.length() && s.compareTo(low) < 0
					|| s.length() == high.length() && s.compareTo(high) > 0)
				return;
			count++;
			return;
		}

		for (char[] p : pairs) {
			ch[left] = p[0];
			ch[right] = p[1];
			if (ch.length != 1 && ch[0] == '0')
				continue;
			if (left < right || left == right && p[0] == p[1]) {
				strobogrammaticUtil(low, high, ch, left + 1, right - 1);
			}
		}

	}

	public int countUnivalSubtrees(TreeNode root) {
		int[] count = { 0 };
		countUnivalSubtreesUtil(root, count);
		return count[0];
	}

	public boolean countUnivalSubtreesUtil(TreeNode root, int[] count) {
		if (root == null)
			return true;
		boolean left = countUnivalSubtreesUtil(root.left, count);
		boolean right = countUnivalSubtreesUtil(root.right, count);
		if (left && right) {
			if (root.left != null && root.left.val != root.val)
				return false;
			if (root.right != null && root.right.val != root.val)
				return false;
			count[0]++;
			return true;
		}
		return false;
	}

	public List<Integer> closestKValues(TreeNode root, double target, int k) {
		Queue<Integer> que = new LinkedList<Integer>();
		Stack<TreeNode> stk = new Stack<TreeNode>();
		TreeNode cur = root;
		while (cur != null) {
			stk.push(cur);
			cur = cur.left;
		}

		while (!stk.isEmpty()) {
			TreeNode top = stk.pop();
			if (que.size() < k)
				que.add(top.val);
			else {
				int first = que.peek();
				if (Math.abs(first - target) > Math.abs(top.val - target)) {
					que.poll();
					que.offer(top.val);
				} else
					break;
			}
			if (top.right != null) {
				top = top.right;
				while (top != null) {
					stk.push(top);
					top = top.left;
				}
			}
		}
		return (List<Integer>) que;
	}

	public List<String> generateAbbreviations(String word) {
		List<String> res = new ArrayList<String>();
		generateAbbrevUtil(word, 0, "", 0, res);
		return res;
	}

	public void generateAbbrevUtil(String word, int cur, String sol, int count,
			List<String> res) {
		if (cur == word.length()) {
			if (count > 0)
				sol += count;
			res.add(sol);
			return;
		} else {
			generateAbbrevUtil(word, cur + 1, sol, count + 1, res);
			generateAbbrevUtil(word, cur + 1, sol + (count > 0 ? count : "")
					+ word.charAt(cur), 0, res);
		}
	}

	/*
	 * google interview 给一个abbrevation 方法定义为: "abcdef" - > "a4f", "bdg" - >
	 * ''b1g"。 给一个word，一个array of words， 写一个function判断这个word的abbrevation是否和list
	 * 任一单词的abbrevation 一样
	 */
	public boolean abbrevationExist(String[] words, String s) {
		int n = s.length();
		if (n < 2)
			return false;
		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			if (word.length() == s.length()) {
				if (word.charAt(0) == s.charAt(0)
						&& word.charAt(n - 1) == s.charAt(n - 1)) {
					return true;
				}
			}
		}
		return false;
	}

	// google interview
	/*
	 * 给一个array of integers， e.g. [1,2,3]。 它的所有permutation可以sort为： [1,2,3],
	 * [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]。写一个function 返回input array 的
	 * nth
	 * permutation。要求不compute任何其他permutation，就是不能算了所有permutation再返回第n个。如果input
	 * array 长度为k，naive 方法要O(k!)， 小哥要O(k)的。
	 */

	public String getKthPermutation(int[] nums, int k) {
		List<Integer> lst = new ArrayList<Integer>();
		int fact = 1;
		for (int i = 0; i < nums.length; i++) {
			lst.add(nums[i]);
			fact *= i + 1;
		}
		int n = nums.length;
		k--;
		StringBuilder res = new StringBuilder();
		for (int i = 0; i < n; i++) {
			fact /= (n - i);
			int index = k / fact;
			res.append(lst.get(index));
			lst.remove(index);
			k %= fact;
		}
		return res.toString();
	}

	public List<List<Integer>> verticalOrder(TreeNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (root == null)
			return res;
		TreeMap<Integer, List<Integer>> map = new TreeMap<Integer, List<Integer>>();
		verticalOrderUtil(root, map, 0);
		for (Map.Entry<Integer, List<Integer>> entry : map.entrySet()) {
			res.add(entry.getValue());
		}
		return res;
	}

	public void verticalOrderUtil(TreeNode root,
			TreeMap<Integer, List<Integer>> map, int index) {
		if (root == null)
			return;
		if (map.containsKey(index)) {
			map.get(index).add(root.val);
		} else {
			List<Integer> lst = new ArrayList<Integer>();
			lst.add(root.val);
			map.put(index, lst);
		}
		verticalOrderUtil(root.left, map, index - 1);
		verticalOrderUtil(root.right, map, index + 1);
	}

	public List<List<Integer>> verticalOrder2(TreeNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		Map<Integer, List<Integer>> map = new HashMap<Integer, List<Integer>>();
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		Queue<Integer> cols = new LinkedList<Integer>();
		int min = 0, max = 0;
		que.offer(root);
		cols.offer(0);
		while (!que.isEmpty()) {
			TreeNode node = que.poll();
			int col = cols.poll();
			if (!map.containsKey(col))
				map.put(col, new ArrayList<Integer>());
			map.get(col).add(node.val);
			if (node.left != null) {
				que.offer(node.left);
				cols.offer(col - 1);
				if (col - 1 < min)
					min = col - 1;
			}

			if (node.right != null) {
				que.offer(node.right);
				cols.offer(col + 1);
				if (col + 1 > max)
					max = col + 1;
			}
		}
		for (int i = min; i <= max; i++) {
			res.add(map.get(i));
		}
		return res;
	}

	// google interview UTF validation
	public boolean validateUTF8(byte[] bytes) {
		int expectedLen;
		if (bytes.length == 0)
			return false;
		// First 8 bits represent the length
		if ((bytes[0] & 0b10000000) == 0b00000000)
			expectedLen = 1;
		if ((bytes[0] & 0b11100000) == 0b11000000)
			expectedLen = 2;
		if ((bytes[0] & 0b11110000) == 0b11100000)
			expectedLen = 3;
		if ((bytes[0] & 0b11111000) == 0b11110000)
			expectedLen = 4;
		if ((bytes[0] & 0b11111100) == 0b11111000)
			expectedLen = 5;
		if ((bytes[0] & 0b11111110) == 0b11111100)
			expectedLen = 6;
		else
			return false;

		if (expectedLen != bytes.length)
			return false;
		// all the following sequence, first two bits begin with "10"
		for (int i = 1; i < bytes.length; i++) {
			if ((bytes[i] & 0b11000000) != 0b10000000)
				return false;
		}
		return true;
	}

	public int countComponents(int n, int[][] edges) {
		int[] ids = new int[n];
		for (int i = 0; i < n; i++) {
			ids[i] = i;
		}

		for (int[] edge : edges) {
			int i = findRoot2(ids, edge[0]);
			int j = findRoot2(ids, edge[1]);
			ids[i] = j;
		}
		int count = 0;
		for (int i = 0; i < n; i++) {
			if (ids[i] == i)
				count++;
		}
		return count;
	}

	public int findRoot2(int[] ids, int i) {
		while (i != ids[i]) {
			ids[i] = ids[ids[i]];
			i = ids[i];
		}
		return i;
	}

	public int countComponents2(int n, int[][] edges) {
		int[] ids = new int[n];
		for (int i = 0; i < n; i++) {
			ids[i] = i;
		}

		for (int[] e : edges) {
			union(ids, e[0], e[1]);
		}
		int count = 0;
		for (int i = 0; i < n; i++) {
			if (ids[i] == i)
				count++;

		}
		return count;
	}

	public void union(int[] ids, int s, int t) {
		if (ids[s] == ids[t])
			return;
		int src = ids[s];
		int dst = ids[t];
		for (int i = 0; i < ids.length; i++) {
			if (ids[i] == src)
				ids[i] = dst;
		}
	}

	// Given one array of length n, create the maximum number of length k.
	public int[] maxArray(int[] nums, int k) {
		int[] res = new int[k];
		int n = nums.length;
		Stack<Integer> stk = new Stack<Integer>();
		for (int i = 0; i < nums.length; i++) {
			while (!stk.isEmpty() && nums[i] > stk.peek()
					&& n - i + stk.size() > k)
				stk.pop();
			if (stk.size() < k)
				stk.push(nums[i]);
		}
		while (k > 0) {
			res[--k] = stk.pop();
		}

		return res;
	}

	public int[] maxArray2(int[] nums, int k) {
		int[] res = new int[k];
		int n = nums.length;
		int j = 0;
		for (int i = 0; i < nums.length; i++) {
			while (n - i + j > k && j > 0 && nums[i] > res[j - 1])
				j--;
			if (j < k)
				res[j++] = nums[i];
		}

		System.out.println(Arrays.toString(res));
		return res;
	}

	// Given two array of length m and n, create maximum number of length k = m
	// + n.

	public int[] merge(int[] nums1, int[] nums2, int k) {
		int[] res = new int[k];
		for (int r = 0, i = 0, j = 0; r < k; r++) {
			res[r] = greater(nums1, i, nums2, j) ? nums1[i++] : nums2[j++];
		}
		return res;
	}

	public boolean greater(int[] nums1, int i, int[] nums2, int j) {
		while (i < nums1.length && j < nums2.length && nums1[i] == nums2[j]) {
			i++;
			j++;
		}
		return j == nums2.length || i < nums1.length && nums1[i] > nums2[j];
	}

	public int[] maxNumber(int[] nums1, int[] nums2, int k) {
		int[] res = new int[k];
		int n = nums1.length, m = nums2.length;
		for (int i = Math.max(0, k - m); i <= k && i <= n; i++) {
			int[] candidate = merge(maxArray(nums1, i), maxArray(nums2, k - i),
					k);
			if (greater(candidate, 0, res, 0))
				res = candidate;
		}
		return res;
	}

	public void reverseWords(char[] s) {
		reverse(s, 0, s.length - 1);
		int start = 0;
		for (int i = 0; i < s.length; i++) {
			if (s[i] == ' ') {
				reverse(s, start, i - 1);
				start = i + 1;
			}
		}
		reverse(s, start, s.length - 1);
		System.out.println(Arrays.toString(s));
	}

	public void reverse(char[] s, int beg, int end) {
		while (beg < end) {
			char c = s[beg];
			s[beg++] = s[end];
			s[end--] = c;
		}
	}

	// read 4k
	public int read(char[] buf) {
		return 4;
	}

	public int read(char[] buf, int n) {
		int total = 0;
		boolean eof = false;
		char[] t = new char[4];
		while (!eof && total < n) {
			int count = read(t);
			eof = count < 4;
			count = Math.min(count, n - total);
			for (int i = 0; i < count; i++) {
				buf[total++] = t[i];
			}
		}
		return total;
	}

	// read n chars multiple times give read4
	Queue<Character> readQue = new LinkedList<Character>();

	public int read2(char[] buf, int n) {
		int total = 0;
		char[] buffer = new char[4];
		while (true) {
			int count = read(buffer);
			for (int i = 0; i < count; i++) {
				readQue.offer(buffer[i]);
			}
			int left = Math.min(n - total, que.size());
			for (int i = 0; i < left; i++) {
				buf[total++] = readQue.poll();
			}
			if (left == 0)
				break;
		}
		return total;
	}

	char[] buff = new char[4];
	int offset = 0;
	int bufSize = 0;

	public int readMutipleTimes(char[] buf, int n) {
		boolean eof = false;
		int total = 0;
		int bytes = 0;
		while (!eof && total < n) {
			if (bufSize == 0) {
				bufSize = read(buff);
				eof = bufSize < 4;
			}

			bytes = Math.min(bufSize, n - total);
			for (int i = 0; i < bytes; i++) {
				buf[total++] = buff[offset + i];
			}
			offset = (offset + bytes) % 4;
			bufSize -= bytes;
		}
		return total;
	}

	public int[] twoSum(int[] numbers, int target) {
		int[] res = { -1, -1 };
		int i = 0, j = numbers.length - 1;
		while (i < j) {
			int sum = numbers[i] + numbers[j];
			if (sum == target) {
				res[0] = i;
				res[1] = j;
			}
			if (sum > target)
				j--;
			else
				i++;
		}
		return res;
	}

	public int[] twoSum2(int[] nums, int target) {
		int[] res = { -1, -1 };
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < nums.length; i++) {
			if (map.containsKey(target - nums[i])) {
				res[0] = map.get(target - nums[i]);
				res[1] = i;
				break;
			}
			map.put(nums[i], i);
		}
		return res;
	}

	public int minTotalDistance4(int[][] grid) {
		int m = grid.length, n = grid[0].length;
		int[] rowSum = new int[m];
		int[] colSum = new int[n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (grid[i][j] == 1) {
					rowSum[i]++;
					colSum[j]++;
				}
			}
		}
		return minDistance(rowSum) + minDistance(colSum);
	}

	public int minDistance(int[] num) {
		int i = -1, j = num.length;
		int d = 0, left = 0, right = 0;
		while (i != j) {
			if (left < right) {
				d += left;
				left += num[++i];
			} else {
				d += right;
				right += num[--j];
			}
		}
		return d;
	}

	public int minPatches(int[] nums, int n) {
		long miss = 1;
		int add = 0, i = 0;
		while (miss <= n) {
			if (i < nums.length && nums[i] <= miss) {
				miss += nums[i++];
			} else {
				miss <<= 1;
				add++;
			}
		}
		return add;
	}

	/*
	 * During building, we record the difference between out degree and in
	 * degree diff = outdegree - indegree. When the next node comes, we then
	 * decrease diff by 1, because the node provides an in degree. If the node
	 * is not null, we increase diff by 2, because it provides two out degrees.
	 */
	public boolean isValidSerialization(String preorder) {
		if (preorder.length() == 0)
			return false;
		String[] nodes = preorder.split(",");
		int diff = 1;
		for (int i = 0; i < nodes.length; i++) {
			if (--diff < 0)
				return false;
			if (!nodes[i].equals("#"))
				diff += 2;
		}
		return diff == 0;
	}

	/*
	 * Longest Zig-Zag Subsequence x1 < x2 > x3 < x4 > x5 < …. xn or x2 > x2 <
	 * x3 > x4 < x5 > …. xn
	 */
	/*
	 * Z[i][0] = Length of the longest Zig-Zag subsequence ending at index i and
	 * last element is greater than its previous element Z[i][1] = Length of the
	 * longest Zig-Zag subsequence ending at index i and last element is smaller
	 * than its previous element
	 */
	public int longestZigzagSequence(int[] A) {
		if (A.length < 2)
			return A.length;
		int n = A.length;
		int[][] dp = new int[n][2];
		for (int i = 0; i < n; i++) {
			dp[i][0] = dp[i][1] = 1;
		}
		int longest = 1;
		for (int i = 1; i < n; i++) {
			for (int j = 0; j < i; j++) {
				if (A[j] < A[i] && dp[j][0] + 1 > dp[i][0])
					dp[i][0] = dp[j][0] + 1;
				if (A[j] > A[i] && dp[j][1] + 1 > dp[j][1])
					dp[i][1] = dp[j][1] + 1;
			}
			longest = Math.max(longest, Math.max(dp[i][0], dp[i][1]));
		}
		return longest;
	}

	/*
	 * Parse XML 现在有一个Tokenizer，返回的Token都是XML标签或者内容，比如(open, html)(inner,
	 * hello)(close, html)表示<html>hello</html>，每一个括号及其内容是一个Token，请问如何表示这个XML文件。
	 */

	public XMLNode parseXML(String s) {
		// 以右括号为delimiter
		StringTokenizer tknz = new StringTokenizer(s, ")");
		Stack<XMLNode> stk = new Stack<XMLNode>();
		XMLNode root = convertToken2Node(tknz.nextToken());
		stk.push(root);

		while (!stk.isEmpty()) {
			if (!tknz.hasMoreTokens())
				break;
			XMLNode node = convertToken2Node(tknz.nextToken());
			XMLNode father = stk.peek();

			switch (node.type) {
			case "open":
				father.children.add(node);
				stk.push(node);
				break;
			case "inner":
				father.children.add(node);
				break;
			case "close":
				stk.pop();
				break;

			}
		}
		return root;
	}

	public XMLNode convertToken2Node(String s) {
		String[] ss = s.substring(1).split(",");
		String type = ss[0];
		String val = ss[1];
		return new XMLNode(type, val);
	}

	public void printXMLTree(XMLNode root, int depth) {
		for (int i = 0; i < depth; i++) {
			System.out.print("-");
		}
		System.out.println(root.type + ":" + root.val);
		for (XMLNode node : root.children) {
			printXMLTree(node, depth + 1);
		}
	}

	/*
	 * Closest leaf to a given node in Binary Tree. Given a Binary Tree and a
	 * node x in it, find distance of the closest leaf to x in Binary Tree. If
	 * given node itself is a leaf, then distance is 0.
	 */

	public int minimumDistance(TreeNode root, TreeNode target) {
		if (root == null)
			return 0;
		int[] min = { 0 };
		findLeafDown(target, 0, min);
		findLeafThroughParent(root, target, min);
		return min[0];
	}

	public void findLeafDown(TreeNode root, int level, int[] min) {
		if (root == null)
			return;
		if (root.left == null && root.right == null) {
			if (level < min[0])
				min[0] = level;
		}
		findLeafDown(root.left, level + 1, min);
		findLeafDown(root.right, level + 1, min);
	}

	public int findLeafThroughParent(TreeNode root, TreeNode target, int[] min) {
		if (root == null)
			return -1;
		if (root == target)
			return 0;
		int l = findLeafThroughParent(root.left, target, min);
		// left subtree has target
		if (l != -1) {
			// Find closest leaf in right subtree
			findLeafDown(root.right, l + 2, min);
			return l + 1;
		}
		int r = findLeafThroughParent(root.right, target, min);
		// right subtree has target
		if (r != -1) {
			// Find closest leaf in left subtree
			findLeafDown(root.left, r + 2, min);
			return r + 1;
		}
		return -1;
	}

	// google interview
	// given sorted arrat, output sorted square array
	public List<Integer> squareArray(int[] nums) {
		int i = 0, j = nums.length - 1;
		int index = -1;
		while (i <= j) {
			int mid = (i + j) / 2;
			if (nums[mid] > 0)
				j = mid - 1;
			else if (nums[mid] < 0)
				i = mid + 1;
			else {
				index = mid;
				break;
			}
		}

		int idx1, idx2;
		if (index == -1) {
			idx1 = i;
			idx2 = i - 1;
		} else {
			idx1 = idx2 = index;
		}
		System.out.println("index is " + idx1 + " " + idx2);
		List<Integer> res = new ArrayList<Integer>();
		while (idx1 < nums.length && idx2 >= 0) {
			if (Math.abs(nums[idx1]) > Math.abs(nums[idx2])) {
				res.add(nums[idx2] * nums[idx2]);
				idx2--;
			} else if (Math.abs(nums[idx1]) < Math.abs(nums[idx2])) {
				res.add(nums[idx1] * nums[idx1]);
				idx1++;
			} else {
				res.add(nums[idx1] * nums[idx1]);
				idx1++;
				idx2--;
			}
		}
		while (idx1 < nums.length) {
			res.add(nums[idx1] * nums[idx1]);
			idx1++;
		}
		while (idx2 >= 0) {
			res.add(nums[idx2] * nums[idx2]);
			idx2--;
		}
		return res;
	}

	// google interview
	// y=a*x^2+b*x+c，给一个sorted list of x， return sorted y。
	public int[] sortSortedArray(int[] arr, int a, int b, int c) {
		int[] res = new int[arr.length];
		int pivot = -b / (2 * a);
		int index = findPivot(arr, pivot);
		int left = index, right = index + 1;
		int pos = 0;
		while (left >= 0 && right < arr.length) {
			if (Math.abs(arr[left] - pivot) < Math.abs(arr[right] - pivot))
				res[pos++] = arr[left--];
			else
				res[pos++] = arr[right++];
		}

		while (left >= 0) {
			res[pos++] = arr[left--];
		}
		while (right < arr.length) {
			res[pos++] = arr[right++];
		}
		for (int i = 0; i < arr.length; i++) {
			res[i] = getRes(res[i], a, b, c);
		}
		if (a < 0)
			reverse(res);
		return res;
	}

	public int getRes(int i, int a, int b, int c) {
		return a * i * i + b * i + c;
	}

	public void reverse(int[] A) {
		int beg = 0, end = A.length - 1;
		while (beg < end) {
			int t = A[beg];
			A[beg] = A[end];
			A[end] = t;
			beg++;
			end--;
		}
	}

	public int findPivot(int[] arr, int pivot) {
		int beg = 0, end = arr.length - 1;

		while (beg + 1 < end) {
			int mid = beg + (end - beg) / 2;
			if (arr[mid] == pivot)
				return mid;
			else if (arr[mid] > pivot)
				end = mid;
			else
				beg = mid;
		}
		if (Math.abs(arr[beg] - pivot) < Math.abs(arr[end] - pivot))
			return beg;
		return end;
	}

	// uber
	/*
	 * 
	 * 
	 * 
	 * Matrix = [
	 * 
	 * [0, 0, 0, 0, 0],
	 * 
	 * [0, 1, 0, 1, 0],
	 * 
	 * [0, 0 , 0, 1, 0] ]
	 * 
	 * 1 is wall.
	 * 
	 * 
	 * Question: check if there is path from the left top to the right bottom?
	 * 
	 * Follow up: if yes, print the path
	 */
	public List<String> printPath(int[][] matrix) {
		List<String> res = new ArrayList<String>();
		int m = matrix.length;
		if (m == 0)
			return res;
		int n = matrix[0].length;
		boolean[][] used = new boolean[m][n];
		printPathUtil(matrix, 0, 0, used, "", res);
		return res;
	}

	public void printPathUtil(int[][] matrix, int i, int j, boolean[][] used,
			String path, List<String> res) {
		if (i == matrix.length - 1 && j == matrix[0].length - 1) {
			res.add(path);
			return;
		}
		if (i < 0 || i >= matrix.length || j < 0 || j >= matrix[0].length
				|| matrix[i][j] == 1 || used[i][j])
			return;
		used[i][j] = true;
		printPathUtil(matrix, i + 1, j, used, path + "D", res);
		printPathUtil(matrix, i - 1, j, used, path + "U", res);
		printPathUtil(matrix, i, j + 1, used, path + "L", res);
		printPathUtil(matrix, i, j - 1, used, path + "R", res);
		used[i][j] = false;
	}

	public List<String> findItinerary(String[][] tickets) {
		List<String> res = new ArrayList<String>();
		int n = tickets.length;
		if (n == 0)
			return res;
		Map<String, PriorityQueue<String>> map = new HashMap<String, PriorityQueue<String>>();
		for (String[] ticket : tickets) {
			if (!map.containsKey(ticket[0]))
				map.put(ticket[0], new PriorityQueue<String>());
			map.get(ticket[0]).offer(ticket[1]);
		}
		dfsFindItinerary("JFK", map, res);
		return res;
	}

	public void dfsFindItinerary(String departure,
			Map<String, PriorityQueue<String>> map, List<String> res) {
		PriorityQueue<String> des = map.get(departure);
		while (des != null && !des.isEmpty()) {
			dfsFindItinerary(des.poll(), map, res);
		}
		res.add(0, departure);
	}

	public List<String> findItineraryIterative(String[][] tickets) {
		List<String> res = new ArrayList<String>();
		Map<String, PriorityQueue<String>> map = new HashMap<String, PriorityQueue<String>>();
		Stack<String> stk = new Stack<String>();

		for (String[] ticket : tickets) {
			if (!map.containsKey(ticket[0])) {
				map.put(ticket[0], new PriorityQueue<String>());
			}
			map.get(ticket[0]).offer(ticket[1]);
		}
		stk.push("JFK");

		while (!stk.isEmpty()) {
			while (map.containsKey(stk.peek())
					&& !map.get(stk.peek()).isEmpty()) {
				stk.push(map.get(stk.peek()).poll());
			}
			res.add(0, stk.pop());
		}
		return res;
	}

	public List<String> findItineraryIterative2(String[][] tickets) {
		List<String> res = new ArrayList<String>();
		Map<String, PriorityQueue<String>> map = new HashMap<String, PriorityQueue<String>>();
		Stack<String> stk = new Stack<String>();

		for (String[] ticket : tickets) {
			if (!map.containsKey(ticket[0])) {
				map.put(ticket[0], new PriorityQueue<String>());
			}
			map.get(ticket[0]).offer(ticket[1]);
		}
		stk.push("JFK");
		while (!stk.isEmpty()) {
			String top = stk.peek();
			if (map.containsKey(top) && !map.get(top).isEmpty()) {
				stk.push(map.get(top).poll());
			} else {
				res.add(0, top);
				stk.pop();
			}
		}
		return res;
	}

	// public boolean findItineraryUtil(String start,
	// Map<String, PriorityQueue<String>> map, List<String> res) {
	// PriorityQueue<String> destinations = map.get(start);
	//
	// for (String des : destinations) {
	// map.get(start).remove(des);
	// if (map.get(start).size() == 0)
	// map.remove(start);
	// if (findItineraryUtil(des, map, count, res, cur + 1))
	// return true;
	// map.put(start, destinations);
	// }
	// return false;
	// }

	/*
	 * google interview 2) Check if all integers in the list can be grouped into
	 * 5 consecutive number. For example [1,2,2,3,3,4,4,5,5,6] should return
	 * True because it can be grouped into (1,2,3,4,5)(2,3,4,5,6) with no other
	 * elements left.
	 */

	public boolean groupNumbers(List<Integer> nums) {
		if (nums.size() % 5 != 0)
			return false;
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int num : nums) {
			if (!map.containsKey(num))
				map.put(num, 1);
			else
				map.put(num, map.get(num) + 1);
		}

		Collections.sort(nums);
		for (int num : nums) {
			int count = 0;
			while (count < 5) {
				if (map.containsKey(num)) {
					map.put(num, map.get(num) - 1);
					if (map.get(num) == 0)
						map.remove(num);
					num++;
				} else {
					break;
				}
			}

		}
		return map.size() == 0;
	}

	/*
	 * You have several coins with values in a bag. You can grab any amount of
	 * coins from the bag once. How many different values of the sum of coins
	 * can you get? Input is the list of values of the coins and output should
	 * be possibilities of sums of coin values.
	 */

	public List<Integer> coinsSums(List<Integer> coins) {
		List<Integer> res = new ArrayList<Integer>();
		if (coins.size() == 0)
			return res;
		for (int i = 0; i <= coins.size(); i++) {
			coinsSumsUtil(coins, i, 0, 0, res, 0);
		}

		return res;
	}

	public void coinsSumsUtil(List<Integer> coins, int size, int cursum,
			int dep, List<Integer> res, int pos) {
		if (dep == size) {
			res.add(cursum);
			return;
		}
		for (int i = pos; i < coins.size(); i++) {
			coinsSumsUtil(coins, size, cursum + coins.get(i), dep + 1, res,
					i + 1);
		}
	}

	public ArrayList<String> letterCombinations(String digits) {
		// Write your code here
		ArrayList<String> res = new ArrayList<String>();
		if (digits.length() == 0)
			return res;
		letterCombinationsUtil(0, digits, "", res);
		return res;
	}

	public void letterCombinationsUtil(int cur, String digits, String sol,
			ArrayList<String> res) {
		if (cur == digits.length()) {
			res.add(sol);
			return;
		}
		String s = getString(digits.charAt(cur) - '0');

		for (int i = 0; i < s.length(); i++) {
			letterCombinationsUtil(cur + 1, digits, sol + s.charAt(i), res);
		}
	}

	public String getString(int i) {
		String[] strs = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs",
				"tuv", "wxyz" };
		return strs[i];
	}

	// google interview
	// find local minimum
	public int findLocalMinimum(int[] A) {
		int left = 0, right = A.length - 1;
		while (left < right) {
			int mid = left + (right - left) / 2;
			if (A[mid] > A[mid + 1])
				left = mid + 1;
			else
				right = mid;
		}
		return left;
	}

	public int findLocalMinimum2(int[] A) {
		return findLocalMinimum(A, 0, A.length - 1);
	}

	public int findLocalMinimum(int[] A, int left, int right) {
		if (left == right)
			return A[left];
		if (left + 1 == right)
			return A[left] < A[right] ? A[left] : A[right];
		int mid = (left + right) / 2;
		if (A[mid] < A[mid - 1] && A[mid] < A[mid + 1])
			return A[mid];
		if (A[mid] > A[mid + 1])
			return findLocalMinimum(A, mid + 1, right);
		else
			return findLocalMinimum(A, left, mid);

	}

	// google interview
	/*
	 * 给一个string，可以删除字符，可以打乱字符，返回一个最长的回文string
	 */
	public String longestPalindromeGoogle(String s) {
		if (s.length() < 2)
			return s;
		int[] chars = new int[256];
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			chars[c - 'a']++;
		}
		StringBuilder sb = new StringBuilder();
		int mid = -1;
		int maxOdd = 0;
		for (int i = 0; i < 256; i++) {
			if (chars[i] != 0 && chars[i] % 2 == 0) {
				for (int j = 0; j < chars[i] / 2; j++) {
					sb.append((char) (i + 'a'));
					sb.insert(0, (char) (i + 'a'));
				}
			} else if (chars[i] > maxOdd) {
				mid = i;
				maxOdd = chars[i];
			}
		}
		for (int j = 0; j < maxOdd; j++) {
			sb.insert(sb.length() / 2, (char) (mid + 'a'));
		}

		return sb.toString();
	}

	// follow up
	public List<String> getAllLongestPanlindromes(String s) {
		if (s.length() < 2)
			return new ArrayList<String>(Arrays.asList(s));
		int[] chars = new int[256];
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			chars[c - 'a']++;
		}
		StringBuilder sb = new StringBuilder();
		Set<Character> set = new HashSet<Character>();
		for (int i = 0; i < 256; i++) {
			if (chars[i] != 0) {
				for (int j = 0; j < chars[i] / 2; j++) {
					sb.append((char) (i + 'a'));
				}
			}
			if (chars[i] % 2 == 1)
				set.add((char) (i + 'a'));
		}

		List<String> res = new ArrayList<String>();
		boolean[] used = new boolean[sb.length()];
		getPalindromesPerms(sb.toString(), new StringBuilder(), used, res, set);
		return res;
	}

	public void getPalindromesPerms(String s, StringBuilder sol,
			boolean[] used, List<String> res, Set<Character> set) {
		if (sol.length() == s.length()) {
			Iterator<Character> it = set.iterator();
			while (it.hasNext()) {
				char c = it.next();
				res.add(sol.toString() + c + sol.reverse().toString());
				sol.reverse();
			}
			return;
		}

		for (int i = 0; i < s.length(); i++) {
			if (!used[i]) {
				if (i != 0 && s.charAt(i) == s.charAt(i - 1) && !used[i - 1])
					continue;
				used[i] = true;
				sol.append(s.charAt(i));
				getPalindromesPerms(s, sol, used, res, set);
				sol.deleteCharAt(sol.length() - 1);
				used[i] = false;
			}
		}
	}

	public List<String> printAllPossiblePalindromes(String str) {
		List<String> res = new ArrayList<String>();
		int[] freq = new int[256];
		if (!isPalin(str, freq))
			return res;
		System.out.println(Arrays.toString(freq));
		StringBuilder sb = new StringBuilder();
		char oddC = ' ';
		for (int i = 0; i < 256; i++) {
			if (freq[i] == 0) {
				continue;
			}
			if (freq[i] % 2 == 1) {
				oddC = (char) (i + 'a');
			}
			for (int j = 0; j < freq[i] / 2; j++) {
				sb.append((char) (i + 'a'));
			}
		}
		boolean[] used = new boolean[sb.length()];
		System.out.println(sb.toString() + " " + (oddC));
		printAllPossiblePalindromesUtil(sb.toString(), used, "", res, oddC);
		return res;
	}

	public void printAllPossiblePalindromesUtil(String s, boolean[] used,
			String sol, List<String> res, char c) {
		if (sol.length() == s.length()) {
			StringBuilder sb = new StringBuilder(s);
			res.add(s + c + sb.reverse().toString());
		}
		for (int i = 0; i < s.length(); i++) {
			if (!used[i]) {
				if (i != 0 && s.charAt(i - 1) == s.charAt(i) && !used[i - 1])
					continue;
				used[i] = true;
				printAllPossiblePalindromesUtil(s, used, sol + s.charAt(i),
						res, c);
				used[i] = false;
			}
		}
	}

	public boolean isPalin(String s, int[] freq) {
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			freq[c - 'a']++;
		}
		int odd = 0;
		for (int i = 0; i < 256; i++) {
			if (freq[i] % 2 == 1)
				odd++;
		}
		return odd <= 1;
	}

	public boolean canSurvive(char[][] board, int x, int y) {
		int m = board.length;
		if (m == 0)
			return false;
		int n = board[0].length;
		Queue<Integer> que = new LinkedList<Integer>();
		que.offer(x * n + y);
		boolean[][] visited = new boolean[m][n];
		visited[x][y] = true;
		while (!que.isEmpty()) {
			int cur = que.poll();
			int row = cur / n;
			int col = cur % n;
			if (board[row][col] == ' ')
				return true;
			if (row + 1 < m && !visited[row + 1][col]
					&& board[row + 1][col] == 'o' || board[row + 1][col] == ' ') {
				que.offer((row + 1) * n + col);
				visited[row + 1][col] = true;
			}
			if (row - 1 >= 0 && !visited[row - 1][col]
					&& board[row - 1][col] == 'o' || board[row - 1][col] == ' ') {
				que.offer((row - 1) * n + col);
				visited[row - 1][col] = true;
			}
			if (col + 1 < n && !visited[row][col + 1]
					&& board[row][col + 1] == 'o' || board[row][col + 1] == ' '
					&& !visited[row][col + 1]) {
				que.offer(row * n + col + 1);
				visited[row][col + 1] = true;
			}
			if (col - 1 >= 0 && !visited[row][col - 1]
					&& board[row][col - 1] == 'o' || board[row][col - 1] == ' ') {
				que.offer(row * n + col - 1);
				visited[row][col - 1] = true;
			}
		}
		return false;
	}

	public int longestOccupyTime(Event[] events) {
		if (events.length == 0)
			return 0;
		Arrays.sort(events, new Comparator<Event>() {

			@Override
			public int compare(Event o1, Event o2) {
				// TODO Auto-generated method stub
				return o1.in - o2.in;
			}

		});
		int longest = events[0].out - events[0].in;
		int start = events[0].in;
		int end = events[0].out;
		for (int i = 1; i < events.length; i++) {
			if (events[i].in <= events[i - 1].out) {
				end = events[i].out;
			} else {
				longest = Math.max(longest, end - start);
				start = events[i].in;
			}
		}
		longest = Math.max(longest, end - start);
		return longest;
	}

	// google interview
	// decode and encode morse code

	public String encodeMorse(String s, Map<String, String> morseCode) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			sb.append(morseCode.get("" + c) + " ");
		}
		return sb.toString();
	}

	public String decodeMorse(String s, Map<String, String> morseCode) {
		StringBuilder sb = new StringBuilder();
		decodeMorse(s, sb, morseCode);
		return sb.toString();

	}

	public boolean decodeMorse(String s, StringBuilder sb,
			Map<String, String> morseCode) {
		if (morseCode.containsKey(s)) {
			sb.append(morseCode.get(s));
			return true;
		}
		for (int i = 1; i < s.length(); i++) {
			String sub = s.substring(0, i);
			if (morseCode.containsKey(sub)) {
				String code = morseCode.get(sub);
				sb.append(code);
				if (decodeMorse(s.substring(i + 1), sb, morseCode))
					return true;
				else
					sb.delete(sb.length() - code.length(), sb.length());
			}
		}
		return false;
	}

	// google interview ads list, profit, time span, max profit
	public int maxAdsProfit(Ads[] ads, int T) {
		if (ads.length == 0)
			return 0;
		Arrays.sort(ads, new Comparator<Ads>() {
			@Override
			public int compare(Ads a1, Ads a2) {
				if (a1.end == a2.end)
					return a1.start - a2.start;
				return a1.end - a2.end;
			}
		});
		int n = ads.length;
		int[][] dp = new int[T + 1][n + 1];

		for (int i = 1; i <= T; i++) {
			for (int j = 1; j <= n; j++) {
				dp[i][j] = dp[i][j - 1];
				if (ads[j - 1].end <= i) {
					dp[i][j] = Math.max(dp[i][j], dp[ads[j - 1].start][j - 1]
							+ ads[j - 1].profit);
				} else
					break;
			}
		}

		return dp[T][n];
	}

	public int maxAdsProfit2(Ads[] ads, int time) {
		int[] profit = new int[time + 1];
		Arrays.sort(ads, new Comparator<Ads>() {
			public int compare(Ads a1, Ads a2) {
				if (a1.end == a2.end) {
					return a1.start - a2.start;
				}
				return a1.end - a2.end;
			}
		});
		for (int i = 1; i <= time; i++) {
			profit[i] = profit[i - 1];
			for (int j = 1; j <= ads.length; j++) {
				if (ads[j - 1].end <= i) {
					profit[i] = Math.max(profit[i], profit[ads[j - 1].start]
							+ ads[j - 1].profit);
				} else {
					break; // others' endTime must be larger than i
				}
			}
		}
		System.out.println(Arrays.toString(profit));
		return profit[time];
	}

	/*
	 * 给一个list 的时间start, end, profits, 让求出来不冲突的最大profit，follow up, 求出最大的组合list
	 */
	public int maxAdsProfit(Ads[] ads) {
		int n = ads.length;
		Arrays.sort(ads, new Comparator<Ads>() {
			@Override
			public int compare(Ads ad1, Ads ad2) {
				if (ad1.end == ad2.end)
					return ad1.start - ad2.start;
				return ad2.end - ad2.end;
			}
		});
		int end = ads[n - 1].end;
		int[] profit = new int[end + 1];

		for (int i = 1; i <= end; i++) {
			profit[i] = profit[i - 1];
			for (int j = 0; j < n; j++) {
				if (ads[j].end <= i) {
					profit[i] = Math.max(profit[i], profit[ads[j].start]
							+ ads[j].profit);
				} else
					break;
			}
		}
		System.out.println(Arrays.toString(profit));
		return profit[end];
	}

	public List<Ads> maxAdsProfitFollowup(Ads[] ads) {
		int n = ads.length;
		Arrays.sort(ads, new Comparator<Ads>() {
			@Override
			public int compare(Ads ad1, Ads ad2) {
				if (ad1.end == ad2.end)
					return ad1.start - ad2.start;
				return ad2.end - ad2.end;
			}
		});
		int end = ads[n - 1].end;
		int[] profit = new int[end + 1];
		Map<Integer, List<Ads>> map = new HashMap<Integer, List<Ads>>();
		for (int i = 1; i <= end; i++) {
			profit[i] = profit[i - 1];
			List<Ads> lst = map.get(i - 1);
			map.put(i, lst);
			for (int j = 0; j < n; j++) {
				if (ads[j].end <= i) {
					if (profit[ads[j].start] + ads[j].profit > profit[i]) {
						profit[i] = profit[ads[j].start] + ads[j].profit;
						lst = map.get(ads[j].start);
						if (lst == null)
							lst = new ArrayList<Ads>();
						List<Ads> cur = new ArrayList<Ads>(lst);
						cur.add(ads[j]);
						map.put(i, cur);
					}
				} else
					break;
			}
		}

		// Iterator<Integer> it = map.keySet().iterator();
		// while (it.hasNext()) {
		// int key = it.next();
		// System.out.println(key + ": " + map.get(key));
		// }
		// System.out.println(Arrays.toString(profit));
		return map.get(end);
	}

	/*
	 * Google interview Weight of subtree
	 * 
	 * id,parent,weight 10,30,1 30,0,10 20,30,2 50,40,3 40,30,4
	 * 
	 * 0 is the assumed root node with weight 0
	 * 
	 * which describes a tree-like structure -- each line is a node, 'parent'
	 * refers to 'id' of another node.
	 * 
	 * Print out, for each node, the total weight of a subtree below this node
	 * (by convention, the weight of a subtree for node X includes the own
	 * weight of X).
	 */

	public void printSubTreeWeight(List<Node> nodes) {
		Map<Integer, Node> nodeIds = new HashMap<Integer, Node>();
		for (Node node : nodes) {
			nodeIds.put(node.id, node);
		}
		Map<Integer, List<Node>> childMap = new HashMap<Integer, List<Node>>();
		for (Node node : nodes) {
			childMap.put(node.id, new ArrayList<Node>());
		}
		for (Node node : nodes) {
			if (childMap.containsKey(node.parent))
				childMap.get(node.parent).add(node);

		}

		HashMap<Node, Integer> weightMap = new HashMap<Node, Integer>();

		for (Node node : nodes) {
			calculateSubTreeWeight(node, childMap, weightMap);
		}

		Iterator<Node> it = weightMap.keySet().iterator();
		while (it.hasNext()) {
			Node node = it.next();
			System.out.println(node.id + ": " + weightMap.get(node));
		}
	}

	public void calculateSubTreeWeight(Node node,
			Map<Integer, List<Node>> childMap, Map<Node, Integer> weightMap) {
		if (childMap.get(node.id).isEmpty()) {// no child
			weightMap.put(node, node.weight);
			return;
		}
		int weight = node.weight;
		for (Node n : childMap.get(node.id)) {
			if (!weightMap.containsKey(n)) {
				calculateSubTreeWeight(n, childMap, weightMap);
			}
			weight += weightMap.get(n);
		}
		weightMap.put(node, weight);
	}

	public Map<Node, Integer> printSubTreeWeight2(List<Node> nodes) {
		Map<Integer, Node> nodeMap = new HashMap<Integer, Node>();
		for (int i = 0; i < nodes.size(); i++) {
			nodeMap.put(nodes.get(i).id, nodes.get(i));
		}

		Map<Integer, ArrayList<Node>> childMap = new HashMap<Integer, ArrayList<Node>>();
		for (int i = 0; i < nodes.size(); i++) {
			childMap.put(nodes.get(i).id, new ArrayList<Node>());
		}

		for (int i = 0; i < nodes.size(); i++) {
			if (childMap.get(nodes.get(i).parent) != null) {
				childMap.get(nodes.get(i).parent).add(nodes.get(i));
			}
		}

		Map<Node, Integer> weightMap = new HashMap<Node, Integer>();
		for (Node n : nodeMap.values()) {
			if (weightMap.get(n) == null) {
				calculateWeight(n, childMap, weightMap);
			}
		}
		Iterator<Node> it = weightMap.keySet().iterator();
		while (it.hasNext()) {
			Node node = it.next();
			System.out.println(node.id + ": " + weightMap.get(node));
		}
		return weightMap;
	}

	public static void calculateWeight(Node n,
			Map<Integer, ArrayList<Node>> childMap, Map<Node, Integer> weightMap) {
		if (childMap.get(n.id).isEmpty()) {// no child
			weightMap.put(n, n.weight);
		} else {
			int weight = n.weight;
			for (Node child : childMap.get(n.id)) {
				if (weightMap.get(child) == null) {
					calculateWeight(child, childMap, weightMap);
				}
				weight += weightMap.get(child);
			}
			weightMap.put(n, weight);
		}
	}

	// google
	// 树形算术表达式 输出字符串形式,
	public String buildExpression(ExpressionTreeNode root) {
		if (root == null)
			return "";
		if (root.left == null && root.right == null)
			return root.symbol;
		if (root.left == null || root.right == null)
			return "";
		StringBuilder sb = new StringBuilder();
		if (root.symbol.equals("*") || root.symbol.equals("/")) {
			if (root.left.symbol.equals("+") || root.left.symbol.equals("-")) {
				sb.append("(").append(buildExpression(root.left)).append(")");
			} else {
				sb.append(buildExpression(root.left));
			}

			sb.append(root.symbol);
			if (root.right.symbol.equals("+") || root.right.symbol.equals("-")) {
				sb.append("(").append(buildExpression(root.right)).append(")");
			} else {
				sb.append(buildExpression(root.right));
			}
		} else {
			sb.append(buildExpression(root.left)).append(root.symbol)
					.append(buildExpression(root.right));
		}

		return sb.toString();
	}

	public int evaluateExpression(ExpressionTreeNode root) {
		if (root == null)
			return 0;
		int res = 0;
		if (root.left == null && root.right == null)
			return Integer.parseInt(root.symbol);
		int left = evaluateExpression(root.left);
		int right = evaluateExpression(root.right);
		switch (root.symbol) {
		case "+":
			res = left + right;
			break;
		case "-":
			res = left - right;
			break;
		case "*":
			res = left * right;
			break;
		case "/":
			res = left / right;
			break;
		}
		return res;
	}

	public boolean isOperator(String s) {
		return s.equals("+") || s.equals("-") || s.equals("*") || s.equals("/");
	}

	public ExpressionTreeNode constructTree(String[] postfix) {
		if (postfix.length == 0)
			return null;
		Stack<ExpressionTreeNode> stk = new Stack<ExpressionTreeNode>();
		for (String s : postfix) {
			if (!isOperator(s)) {
				ExpressionTreeNode node = new ExpressionTreeNode(s);
				stk.push(node);
			} else {
				ExpressionTreeNode node = new ExpressionTreeNode(s);
				ExpressionTreeNode node1 = stk.pop();
				ExpressionTreeNode node2 = stk.pop();
				node.right = node1;
				node.left = node2;
				stk.push(node);
			}
		}
		return stk.pop();
	}

	public void inorder(ExpressionTreeNode root) {
		if (root == null)
			return;
		inorder(root.left);
		System.out.print(root.symbol);
		inorder(root.right);
	}

	// Count distinct elements in every window of size k
	public void countDistinct(int[] nums, int k) {
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < k; i++) {
			if (!map.containsKey(nums[i])) {
				map.put(nums[i], 1);
			} else {
				map.put(nums[i], map.get(nums[i]) + 1);
			}
		}
		System.out.println(map.size());

		for (int i = k; i < nums.length; i++) {
			int num = nums[i - k];
			if (map.get(num) == 1) {
				map.remove(num);
			} else {
				map.put(num, map.get(num) - 1);
			}
			if (!map.containsKey(nums[i])) {
				map.put(nums[i], 1);
			} else {
				map.put(nums[i], map.get(nums[i]) + 1);
			}
			System.out.println(map.size());
		}
	}

	public Set<Integer> friendCirlce(String[] friends) {
		int n = friends.length;
		char[][] relation = new char[n][n];
		for (int i = 0; i < n; i++) {
			relation[i] = friends[i].toCharArray();
		}
		int[] ids = new int[n];
		for (int i = 0; i < n; i++) {
			ids[i] = i;
		}

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (relation[i][j] == 'y') {
					int id1 = findRoot3(ids, i);
					int id2 = findRoot3(ids, j);
					ids[id2] = id1;
				}
			}
		}
		System.out.println(Arrays.toString(ids));
		Set<Integer> circles = new HashSet<Integer>();
		for (int i = 0; i < ids.length; i++) {
			circles.add(ids[i]);
		}
		return circles;
	}

	public int findRoot3(int[] ids, int i) {
		while (i != ids[i]) {
			ids[i] = ids[ids[i]];
			i = ids[i];
		}
		return i;
	}

	public boolean increasingTriplet(int[] nums) {
		int first = Integer.MAX_VALUE, second = Integer.MAX_VALUE;
		for (int num : nums) {
			if (num <= first)
				first = num;
			else if (num <= second)
				second = num;
			else
				return true;
		}
		return false;
	}

	public int friendCirclesBFS(String[] friends) {
		Queue<Integer> que = new LinkedList<Integer>();
		int n = friends.length;
		boolean[] visited = new boolean[n];
		que.offer(0);
		visited[0] = true;
		int circles = 0;

		while (!que.isEmpty()) {
			int first = que.poll();
			String s = friends[first];
			for (int i = 0; i < s.length(); i++) {
				if (i != first && s.charAt(i) == 'y' && !visited[i]) {
					que.offer(i);
					visited[i] = true;
				}
			}
			if (que.isEmpty()) {
				circles++;
				for (int i = 1; i < friends.length; i++) {
					if (!visited[i]) {
						que.offer(i);
						visited[i] = true;
						break;
					}
				}
			}
		}
		return circles;
	}

	public int longestChain(String[] strs) {
		Set<String> dict = new HashSet<String>();
		for (String s : strs) {
			dict.add(s);
		}
		int longest = 0;
		Map<String, Integer> map = new HashMap<String, Integer>();
		for (String s : strs) {
			int len = findLongestChain(s, dict, map);
			// map.put(s, len);
			longest = Math.max(len, longest);
		}
		return longest;
	}

	public int findLongestChain(String word, Set<String> dict,
			Map<String, Integer> map) {
		if (dict == null)
			return 0;
		if (word.length() == 1)
			return 1;
		if (map.containsKey(word))
			return map.get(word);
		int maxLen = 1;
		for (int i = 0; i < word.length(); i++) {
			StringBuilder sb = new StringBuilder(word);
			sb.deleteCharAt(i);
			String s = sb.toString();
			if (dict.contains(s)) {
				int len = findLongestChain(s, dict, map);
				if (len + 1 > maxLen)
					maxLen = len + 1;
			}
		}
		map.put(word, maxLen);
		return maxLen;
	}

	// 给一个string和一个character set，要求给出所有的可能的string，把在character set
	// 里的string里的char变成大写或者保持小写
	// 比如airplane ｛a,p} 输出{airpalne，Airplane, AirPlane,AirplAne, airPlane, ...}
	public List<String> generateAllPossible(String s, Set<Character> set) {
		List<String> res = new ArrayList<String>();
		generateAllPossibleUtil(0, s, set, res, "");
		return res;
	}

	public void generateAllPossibleUtil(int cur, String s, Set<Character> set,
			List<String> res, String sol) {
		if (sol.length() == s.length()) {
			res.add(sol);
			return;
		}

		for (int i = cur; i < s.length(); i++) {
			char c = s.charAt(i);
			if (set.contains(c)) {
				generateAllPossibleUtil(i + 1, s, set, res,
						sol + Character.toUpperCase(c));
			}
			generateAllPossibleUtil(i + 1, s, set, res, sol + c);

		}
	}

	public List<String> generateAllPossible2(String s, Set<Character> set) {
		List<String> res = new ArrayList<String>();
		Set<Character> visited = new HashSet<Character>();
		generateAllPossibleUtil2(0, s, set, res, "", visited);
		return res;
	}

	public void generateAllPossibleUtil2(int cur, String s, Set<Character> set,
			List<String> res, String sol, Set<Character> visited) {
		if (sol.length() == s.length()) {
			if (!res.contains(sol))
				res.add(sol);
			return;
		}

		for (int i = cur; i < s.length(); i++) {
			char c = s.charAt(i);
			if (set.contains(c)) {
				if (visited.add(c)) {
					generateAllPossibleUtil2(i + 1,
							s.replace(c, Character.toUpperCase(c)), set, res,
							sol.replace(c, Character.toUpperCase(c))
									+ Character.toUpperCase(c), visited);
					generateAllPossibleUtil2(i + 1, s, set, res, sol + c,
							visited);

				} else {
					visited.remove(c);
					return;
				}
			}
			generateAllPossibleUtil2(i + 1, s, set, res, sol + c, visited);
		}
	}

	public List<String> generataAllStringFollowUp(String input,
			Set<Character> set) {
		List<String> res = new ArrayList<String>();
		if (input == null || input.length() == 0) {
			return res;
		}
		HashMap<Character, Character> map = new HashMap<Character, Character>();
		helper(res, new StringBuilder(), set, input, 0, map);
		return res;
	}

	public void helper(List<String> res, StringBuilder sb, Set<Character> set,
			String input, int pos, HashMap<Character, Character> map) {
		if (pos == input.length()) {
			res.add(sb.toString());
			return;
		}
		char c = input.charAt(pos);
		if (!set.contains(c)) {
			sb.append(c);
			helper(res, sb, set, input, pos + 1, map);
			sb.deleteCharAt(sb.length() - 1);
		} else {
			if (map.containsKey(c)) {
				char ch = map.get(c);
				sb.append(ch);
				helper(res, sb, set, input, pos + 1, map);
				sb.deleteCharAt(sb.length() - 1);
			} else {
				sb.append(c);
				map.put(c, c);
				helper(res, sb, set, input, pos + 1, map);
				sb.deleteCharAt(sb.length() - 1);
				map.remove(c);
				char capital = (char) (c - 'a' + 'A');
				map.put(c, capital);
				sb.append(capital);
				helper(res, sb, set, input, pos + 1, map);
				sb.deleteCharAt(sb.length() - 1);
				map.remove(c);
			}
		}
	}

	public List<String> stringCombination(String s, int k) {
		List<String> res = new ArrayList<String>();
		if (s.length() <= k) {
			res.add(s);
			return res;
		}

		stringCombinationUtil(0, s, k, "", res, 0);
		return res;
	}

	public void stringCombinationUtil(int dep, String s, int k, String sol,
			List<String> res, int cur) {
		if (dep == k) {
			res.add(sol);
			return;
		}

		for (int i = cur; i < s.length(); i++) {
			stringCombinationUtil(dep + 1, s, k, sol + s.charAt(i), res, i + 1);
		}
	}

	public static Set<String> generateBigFollowUp(String input,
			Set<Character> set) {
		Set<String> st = new HashSet<String>();
		Set<Character> alreadyAdded = new HashSet<Character>();
		String tempStr = input;
		helperGeneratorFollowUp(st, tempStr, input, set, alreadyAdded);
		return st;
	}

	public static void helperGeneratorFollowUp(Set<String> res, String sb,
			String word, Set<Character> set, Set<Character> added) {
		Iterator<Character> itr = set.iterator();
		res.add(sb);
		while (itr.hasNext()) {
			char temp = itr.next();
			if (added.contains(temp)) {
				added.remove(temp);
				return;
			}
			StringBuilder sb1 = new StringBuilder(sb);
			for (int i = 0; i < word.length(); i++) {
				if (word.charAt(i) == temp) {
					sb1.setCharAt(i, Character.toUpperCase(word.charAt(i)));
					added.add(word.charAt(i));
				}
			}
			helperGeneratorFollowUp(res, sb1.toString(), word, set, added);
		}
	}

	// google OA delete a digit
	public int getLargestNumDeleteOneDigit(int X) {
		if (X <= 1 || X > 1000000000)
			return 0;
		String s = String.valueOf(X);
		int max = 0;
		char last = s.charAt(0);
		for (int i = 1; i < s.length(); i++) {
			char cur = s.charAt(i);
			if (cur != last)
				last = cur;
			else {
				String sub = s.substring(0, i) + s.substring(i + 1);
				int num = Integer.parseInt(sub);
				// System.out.println(num);
				max = Math.max(num, max);
			}
		}
		return max;
	}

	public int longestImagePath(String s) {
		if (s.length() == 0)
			return 0;
		s = s.trim();
		String[] files = s.split("\n");
		int curlen = 0;
		int max = 0;
		Stack<FDGoogle> stk = new Stack<FDGoogle>();
		for (String file : files) {

			FDGoogle fd = new FDGoogle(file);
			System.out.println("file is " + file + " " + fd.isImage);
			if (fd.isImage) {
				max = Math.max(max, curlen + fd.fileName.trim().length());
			} else if (!file.contains(".")) {
				if (stk.isEmpty() || fd.numOfSpaces > stk.peek().numOfSpaces) {
					stk.push(fd);
					System.out.println("pre len is " + curlen);
					curlen += fd.fileName.trim().length() + 1;
					System.out.println("cur len is " + curlen);
				} else {
					while (!stk.isEmpty()
							&& fd.numOfSpaces <= stk.peek().numOfSpaces) {
						curlen -= stk.pop().fileName.trim().length() + 1;
					}
					stk.push(fd);
					curlen += fd.fileName.trim().length() + 1;
				}
			}
		}
		return max;
	}

	public Set<Integer> getAllFactors(int n) {
		Set<Integer> res = new HashSet<Integer>();
		getAllFactorsUtil(n, res, 2);
		return res;
	}

	public void getAllFactorsUtil(int n, Set<Integer> res, int start) {
		if (n % start == 0 && n != start) {
			res.add(start);
		}

		for (int i = start; i < n; i++) {
			if (n % i == 0) {
				res.add(i);
				getAllFactorsUtil(n / i, res, i);
			}
		}
	}

	public int findMissingNumber(String s) {
		for (int i = 1; i < 5; i++) {
			if (identifyStartDigit(s, 0, i)) {
				int res = findMissingNumberUtil(s, i);
				if (res != -1)
					return res;
			}
		}
		return -1;
	}

	public int findMissingNumberUtil(String s, int start) {
		if (start > s.length())
			return -1;
		int num = Integer.parseInt(s.substring(0, start));
		int nextNum1 = num + 1;
		int nextNum2 = num + 2;

		if (s.substring(start).startsWith(nextNum1 + ""))
			return findMissingNumberUtil(s.substring(start),
					String.valueOf(nextNum1).length());
		else if (s.substring(start).startsWith(nextNum2 + ""))
			return nextNum1;
		else
			return -1;
	}

	public boolean identifyStartDigit(String s, int beg, int offset) {
		int num = Integer.parseInt(s.substring(beg, offset));
		int nextNum1 = num + 1;
		int nextNum2 = num + 2;

		if (s.substring(offset).startsWith(nextNum1 + "")
				|| s.substring(offset).startsWith(nextNum2 + ""))
			return true;
		else
			return false;
	}

	public int findMissingNumberGivenString(String s) {
		for (int i = 1; i < s.length() / 2; i++) {
			int preNum = getNumber(s, 0, i);
			int offset = i;
			System.out.println("offset is " + offset);
			while (true) {
				int nextNum1 = preNum + 1;
				int len1 = String.valueOf(nextNum1).length();
				int nextNum2 = preNum + 2;
				int len2 = String.valueOf(nextNum2).length();
				System.out.println(preNum + " " + nextNum1 + " " + nextNum2);
				if (offset + len1 > s.length())
					break;

				int nextNum = getNumber(s, offset, len1);
				System.out.println("nextNum-1 is " + nextNum);
				if (nextNum == nextNum1) {
					preNum = nextNum;
					offset += len1;
					if (offset >= s.length())
						break;
					continue;
				}

				if (offset + len2 > s.length())
					break;
				nextNum = getNumber(s, offset, len2);
				System.out.println("nextNum-2 is " + nextNum);
				if (nextNum == nextNum2) {
					return nextNum1;
				}

				break;// wrong sequence
			}

		}
		return -1;
	}

	public int getNumber(String s, int i, int j) {
		return Integer.parseInt(s.substring(i, i + j));
	}

	public void quicksort(int[] A) {
		if (A.length < 2)
			return;
		quicksort(A, 0, A.length - 1);
		System.out.println(Arrays.toString(A));
	}

	public void quicksort(int[] A, int lo, int hi) {
		if (hi <= lo)
			return;
		int idx = partition(A, lo, hi);
		quicksort(A, lo, idx - 1);
		quicksort(A, idx + 1, hi);
	}

	public int partition(int[] A, int lo, int hi) {
		int left = lo, right = hi;
		int pivot = A[hi];

		while (left < right) {
			if (A[left++] > pivot) {
				swap(A, --left, --right);
			}
		}
		swap(A, left, hi);
		return right;
	}

	// max sum submatrix
	// use kadane as helper
	// // Find sum of every mini-row between left and right columns and save it
	// into temp[]

	public int findMaxSubMatrixSum(int[][] matrix) {
		int m = matrix.length, n = matrix[0].length;
		int max = 0;
		int l = 0, r = 0, t = 0, b = 0;
		for (int left = 0; left < n; left++) {
			int[] temp = new int[m];
			for (int right = left; right < n; right++) {
				for (int i = 0; i < m; i++) {
					temp[i] += matrix[i][right];
				}
				int cursum = kadane(temp);
				if (cursum > max) {
					max = cursum;
					l = left;
					r = right;
					t = top;
					b = bottom;
				}
			}
		}
		System.out.println(t + " " + l + " " + b + " " + r);
		return max;
	}

	int top = -1, bottom = -1;

	public int kadane(int[] A) {
		int sum = 0, max = 0;
		int localtop = 0;
		for (int i = 0; i < A.length; i++) {
			sum += A[i];
			if (sum > max) {
				max = sum;
				top = localtop;
				bottom = i;
			}
			if (sum < 0) {
				sum = 0;
				localtop = i + 1;
			}
		}
		if (max == 0) {
			max = A[0];
			for (int i = 1; i < A.length; i++) {
				if (A[i] > max) {
					max = A[i];
					top = bottom = i;
				}
			}
		}
		return max;
	}

	/*
	 * google interview
	 */
	// Return sum of integer 1..N, excluding multipliers of any single integers
	// in X.
	// Example: N = 10, X = {2}, return 1 + 3 + 5 + 7 + 9 = 25.
	// Example 2: N = 10, X = {2, 3}, return 1 + 5 + 7 = 13.
	// Assume 1 < N <= 2^31. 1 <= K = X.size() <= 10. X contains only distinct
	// positive prime numbers.

	public int getSumExcludePrimes(int n, int[] X) {
		boolean[] primes = new boolean[n];
		for (int i = 0; i < n; i++) {
			primes[i] = true;
		}

		for (int x : X) {
			if (primes[x]) {
				for (int j = x; j < n; j += x)
					primes[j] = false;
			}
		}

		int sum = 0;
		for (int i = 0; i < n; i++) {
			if (primes[i]) {
				sum += i;
			}
		}
		return sum;
	}

	public List<String> getAllPermutations(String s, int k) {
		List<String> res = new ArrayList<String>();
		char[] chars = s.toCharArray();
		Arrays.sort(chars);
		boolean[] used = new boolean[chars.length];
		getAllPermutationsUtil(k, chars, used, new StringBuilder(), res, 0);
		return res;
	}

	public void getAllPermutationsUtil(int max, char[] chars, boolean[] used,
			StringBuilder sol, List<String> res, int cur) {
		if (sol.length() == max) {
			res.add(sol.toString());
		}

		for (int i = 0; i < chars.length; i++) {
			if (i != 0 && !used[i - 1] && chars[i] == chars[i - 1])
				continue;
			if (!used[i]) {
				used[i] = true;
				sol.append(chars[i]);
				getAllPermutationsUtil(max, chars, used, sol, res, i + 1);
				used[i] = false;
				sol.deleteCharAt(sol.length() - 1);
			}
		}
	}

	/*
	 * Google interview Considering there are N engineers in one company with id
	 * from 1 to N, Each of the Engineer has a skill rate R_i, we want to select
	 * one group of engineering whose ids have to be continuous. The size of one
	 * group is the number of engineers in it and the skill rating of the group
	 * is the lowest rating of all the members in the group. Now given N and
	 * array R, for each group from size X from 1 to N, we want to know the
	 * highest skill rating of all the groups whose size is X
	 */
	class SkillRating {
		int id;
		int rating;

		public SkillRating(int id, int rating) {
			this.id = id;
			this.rating = rating;
		}
	}

	public int highestSkillRating(int N, int[] R, int X) {
		PriorityQueue<SkillRating> que = new PriorityQueue<SkillRating>(X,
				new Comparator<SkillRating>() {
					@Override
					public int compare(SkillRating sr1, SkillRating sr2) {
						return sr1.rating - sr2.rating;
					}
				});
		for (int i = 0; i < X; i++) {
			que.offer(new SkillRating(i, R[i]));
		}
		int max = 0;
		for (int i = X; i < N; i++) {
			SkillRating sr = que.peek();
			max = Math.max(max, sr.rating);
			while (!que.isEmpty() && i - sr.id >= X) {
				que.poll();
				sr = que.peek();
			}
			que.offer(new SkillRating(i, R[i]));
		}
		return max;
	}

	/*
	 * Google interview 给一个整数数组（可正可负可重复），返回其中任意两个元素乘积的最大值。
	 */
	public int maxProductOfTwo(int[] A) {
		int max = Integer.MIN_VALUE, secMax = Integer.MIN_VALUE;
		int min = Integer.MAX_VALUE, secMin = Integer.MAX_VALUE;
		for (int i = 0; i < A.length; i++) {
			if (A[i] > max) {
				secMax = max;
				max = A[i];
			} else if (A[i] > secMax) {
				secMax = A[i];
			}
			if (A[i] < min) {
				secMin = min;
				min = A[i];
			} else if (A[i] < secMin) {
				secMin = A[i];
			}
		}
		System.out.println(max + " " + secMax + " " + min + " " + secMin);
		int max1 = max * secMax;
		int max2 = min * secMin;
		return Math.max(max1, max2);
	}

	/*
	 * follow up k max products of two elements in the array
	 */
	public List<Integer> KMaxProductOfTwo(int[] A, int k) {
		PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();
		PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(k,
				new Comparator<Integer>() {
					@Override
					public int compare(Integer i, Integer j) {
						return j - i;
					}
				});

		for (int i = 0; i < A.length; i++) {
			if (maxHeap.size() < k) {
				maxHeap.offer(A[i]);
			} else {
				if (A[i] < maxHeap.peek()) {
					maxHeap.poll();
					maxHeap.offer(A[i]);
				}
			}

			if (minHeap.size() < k) {
				minHeap.offer(A[i]);
			} else {
				if (A[i] > minHeap.peek()) {
					minHeap.poll();
					minHeap.offer(A[i]);
				}
			}
		}

		PriorityQueue<Integer> minHeap2 = new PriorityQueue<Integer>();
		PriorityQueue<Integer> maxHeap2 = new PriorityQueue<Integer>(k,
				new Comparator<Integer>() {
					@Override
					public int compare(Integer i, Integer j) {
						return j - i;
					}
				});

		while (!maxHeap.isEmpty()) {
			minHeap2.offer(maxHeap.poll());
			maxHeap2.offer(minHeap.poll());
		}

		List<Integer> res = new ArrayList<Integer>();
		int max = maxHeap2.poll();
		int min = minHeap2.poll();
		while (!minHeap2.isEmpty() && !maxHeap2.isEmpty()) {
			int secMax = maxHeap2.peek();
			int secMin = minHeap2.peek();
			if (max * secMax > min * secMin) {
				res.add(max * secMax);
				maxHeap2.poll();
			} else {
				res.add(min * secMin);
				minHeap2.poll();
			}
			if (res.size() == k)
				break;
		}
		return res;
	}

	// unique word abbreviation
	public List<String> wordAbbreviations(String word) {
		List<String> res = new ArrayList<String>();
		if (word.length() == 0)
			return res;
		wordAbbreviationsUtil(word, 0, "", 0, res);
		return res;
	}

	public void wordAbbreviationsUtil(String word, int cur, String sol,
			int count, List<String> res) {
		if (cur == word.length()) {
			if (count > 0) {
				sol += count;
			}
			res.add(sol);
		} else {
			wordAbbreviationsUtil(word, cur + 1, sol, count + 1, res);
			wordAbbreviationsUtil(
					word,
					cur + 1,
					count > 0 ? sol + count + word.charAt(cur) : sol
							+ word.charAt(cur), 0, res);
		}
	}

	// airbnb
	// find all pairs that concatenated to be aplindrome
	public List<String> getPalindromaticPairs(String[] input) {
		List<String> res = new ArrayList<String>();
		if (input.length < 2)
			return res;
		Map<String, List<Integer>> map = new HashMap<String, List<Integer>>();
		for (int i = 0; i < input.length; i++) {
			String s = reverseStr(input[i]);
			if (!map.containsKey(s))
				map.put(s, new ArrayList<Integer>());
			map.get(s).add(i);
		}

		for (int i = 0; i < input.length; i++) {
			String s = input[i];
			for (int j = 0; j <= s.length(); j++) {
				String prefix = s.substring(0, j);
				String post = s.substring(j);
				if (map.containsKey(prefix) && isPalindrome(post)) {
					if (map.get(prefix).size() > 1
							|| map.get(prefix).get(0) != i) {
						String p = s + reverseStr(prefix);
						if (!res.contains(p))
							res.add(p);
					}
				}
			}

			for (int j = s.length() - 1; j >= 0; j--) {
				String post = s.substring(j);
				String prefix = s.substring(0, j);
				if (map.containsKey(post) && isPalindrome(prefix)) {
					if (map.get(post).size() > 1 || map.get(post).get(0) != i) {
						String p = reverseStr(post) + s;
						if (!res.contains(p))
							res.add(p);
					}
				}
			}
		}
		return res;
	}

	public String reverseStr(String s) {
		StringBuilder sb = new StringBuilder(s);
		return sb.reverse().toString();
	}

	public boolean isPanlindrome(String s) {
		if (s.length() < 2)
			return true;
		int i = 0, j = s.length() - 1;
		while (i < j) {
			if (s.charAt(i++) != s.charAt(j--))
				return false;
		}
		return true;
	}

	// airbnb
	// find intervals all employees are free
	public Interval findFreeInterval(Interval[] intervals) {
		if (intervals.length == 0)
			return null;
		Arrays.sort(intervals, new Comparator<Interval>() {
			@Override
			public int compare(Interval i1, Interval i2) {
				return i1.start - i2.start;
			}
		});

		Interval start = intervals[0];
		for (int i = 1; i < intervals.length; i++) {
			Interval interv = intervals[i];
			if (start.end < interv.start) {
				// if find the first one
				return new Interval(start.end, interv.start);
				// if find all free intervals
				// System.out.println(new Interval(start.end, interv.start));
				// start=interv;
			} else {
				start.end = Math.max(start.end, interv.end);
			}
		}
		return null;
	}

	/*
	 * 给一个数组代表reservation request，suppose start date, end date back to back.
	 * 比如[5,1,1,5]代表如下预定： Jul 1-Jul6 Jul6-Jul7 Jul7-Jul8 jul8-Jul13
	 * 当然最开始那个Jul1是随便选就好的啦。 现在input的意义搞清楚了。还有一个限制，就是退房跟开始不能是同一天，比如如果接了Jul
	 * 1-Jul6，Jul6-Jul7就不能接了。那问题就是给你个数组，算算最多能把房子租出去多少天。这个例子的话就是10天. [4,9,6]=10.
	 * 
	 * [4,10,3,1,5]=1
	 */

	public int getMaxRentalDays(int[] nums) {
		int n = nums.length;
		if (n == 0)
			return 0;
		int[] dp = new int[n];
		dp[0] = nums[0];
		if (n > 1) {
			dp[1] = Math.max(nums[0], nums[1]);
		}
		for (int i = 2; i < nums.length; i++) {
			dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
		}
		return dp[nums.length - 1];
		// int prev2 = 0;
		// int prev1 = nums[0];
		//
		// for (int i = 2; i <= nums.length; i++) {
		// int curr = Math.max(prev1, prev2 + nums[i - 1]);
		//
		// prev2 = prev1;
		// prev1 = curr;
		// }
		//
		// return prev1;
	}

	/*
	 * Airbnb interview Given an array of numbers A = [x1, x2, ..., xn] and T =
	 * Round(x1+x2+... +xn). We want to find a way to round each element in A
	 * such that after rounding we get a new array B = [y1, y2, ...., yn] such
	 * that y1+y2+...+yn = T where yi = Floor(xi) or Ceil(xi), ceiling or floor
	 * of xi.. We also want to minimize sum |x_i-y_i|
	 */

	public int[] rounding(double[] A) {
		int n = A.length;
		int[] res = new int[n];
		int sRound = 0; // seperately rounding
		double sum = 0;
		List<Double> lst = new ArrayList<Double>();
		for (double num : A) {
			lst.add(num);
			sum += num;
			sRound += Math.round(num);
		}
		int tRound = (int) Math.round(sum); // together rounding
		int diff = tRound - sRound;
		boolean over = true;

		if (diff >= 0) {
			Collections.sort(lst, new Comparator<Double>() {
				@Override
				public int compare(Double num1, Double num2) {
					// num1=Math.ceil(num1)-num1;
					// num2=Math.ceil(num2)-num2;
					num1 = num1 - Math.floor(num1);
					num2 = num2 - Math.floor(num2);
					if (num1 > num2)
						return -1;
					else
						return 1;
				}
			});
		} else {
			Collections.sort(lst, new Comparator<Double>() {
				@Override
				public int compare(Double num1, Double num2) {
					// num1=num1-Math.floor(num1);
					// num2=num2-Math.floor(num2);
					num1 = Math.ceil(num1) - num1;
					num2 = Math.ceil(num2) - num2;
					if (num1 > num2)
						return -1;
					else
						return 1;
				}
			});
			over = false;
		}

		diff = Math.abs(diff);
		HashSet<Double> set = new HashSet<Double>();
		for (int i = 0; i < diff; i++) {
			set.add(lst.get(i));
		}

		for (int i = 0; i < A.length; i++) {
			double d = A[i];
			int tmp = (int) Math.round(d);
			if (set.contains(d) && diff > 0) {
				if (over)
					tmp++;
				else
					tmp--;
				diff--;
			}
			res[i] = tmp;
		}
		return res;
	}

	/*
	 * 给你一个list of posts，每个post对应一个host，这个list是已经排序好了的。
	 * 因为同一个host可以发好几个post，用户不希望看到Airbnb给的推荐房源都是来自同一个户主。
	 * 所以面试官希望对这个list调整一下排序，让每一页里的post不出现相同的host，otherwise preserve the
	 * ordering。 每一页中有12个post
	 */

	public void displayPages(List<String> posts) {
		if (posts.size() == 0)
			return;
		Iterator<String> it = posts.iterator();
		int pageNum = 1;
		System.out.println("Page " + pageNum);

		Set<String> visited = new HashSet<String>();
		while (it.hasNext()) {
			String post = it.next();
			String hostId = post.split(",")[0];
			if (visited.add(hostId)) {
				System.out.println(post);
				it.remove();
			}
			if (visited.size() == 12 || !it.hasNext()) {
				visited.clear();
				it = posts.iterator();
				if (posts.size() > 0) {
					pageNum++;
					System.out.println("Page " + pageNum);
				}
			}
		}
	}

	public TreeNode removeHalfNodes(TreeNode root) {
		if (root == null)
			return null;
		root.left = removeHalfNodes(root.left);
		root.right = removeHalfNodes(root.right);

		if (root.left == null && root.right == null)
			return root;
		if (root.left == null) {
			TreeNode newNode = root.right;
			return newNode;
		}
		if (root.right == null) {
			TreeNode newNode = root.left;
			return newNode;
		}
		return root;
	}

	/*
	 * 实现一个mini parser, 输入是以下格式的string:"324"
	 * or"[123,456,[788,799,833],[[]],10,[]]" 要求输出:324 or
	 * [123,456,[788,799,833],[[]],10,[]]
	 */
	// GOTO NestedIntList

	/*
	 * 给一个log file，里面记录两类函数: start(int processId, int startTime)和finish(int
	 * processId, int endTime)，分别记录系统调用某个进程的开始时间和结束时间以及该进程的ID。
	 * 给了2个条件：（1）时间是递增的，（2）对于每个进程，每一个start总有一个对应的finish。 问题是：当遇到一个finish(int
	 * proc1, int time1)函数时，如果有排在proc1之前还有进程没有结束的话（i.e.,
	 * 开始时间在proc1之前的进程），就不打印任何内容；否则打印出所有已经结束的进程，并且按照他们的开始时间顺序打印。 比如： -start(1,1)
	 * -start(2,2) -start(3,3) -end(2,4) -end(3,5) -end(1,6)
	 * 遇到end(2,4)和end(3,5)时不打印，因为进程1的开始时间在进程2和进程3之前，直到遇到end(1,6)时，才能打印，打印顺序如下:
	 * 1: 1, 6 2: 2, 4 3: 3, 5。 分别是进程ID：开始时间，结束时间。
	 */

	public void printProcessOrder(Point[] starts, Point[] ends) {
		Stack<Process> stk = new Stack<Process>();
		Queue<Process> que = new LinkedList<Process>();

		int i = 0, j = 0;
		while (i < starts.length || j < ends.length) {
			if (i < starts.length && starts[i].y < ends[j].y) {
				Process p = new Process(starts[i].x, starts[i].y);
				stk.push(p);
				que.offer(p);
				i++;
			} else {
				if (ends[j].x == stk.peek().pid) {
					Process top = stk.pop();
					que.remove(top);
					System.out.println(ends[j].x + ": " + top.start + ", "
							+ ends[j].y);
				} else {
					Process p = new Process(ends[j].x, ends[j].y);
					stk.push(p);
				}
				j++;
			}
		}

		Stack<Process> tStk = new Stack<Process>();

		while (!que.isEmpty()) {
			while (que.peek().pid != stk.peek().pid) {
				tStk.push(stk.pop());
			}
			Process p = que.poll();
			System.out.println(p.pid + ": " + p.start + ", " + stk.pop().start);
			while (!tStk.isEmpty()) {
				stk.push(tStk.pop());
			}
		}
	}

	/*
	 * google interview '*' can match zero or any characters
	 */
	public boolean matching(String s, String p) {
		if (p.length() == 0)
			return s.length() == 0;
		// if(p.equals("*"))
		// return true;
		int i = 0, j = 0;
		while (i < s.length() && j < p.length()) {
			if (s.charAt(i) == p.charAt(j)) {
				i++;
				j++;
			} else if (p.charAt(j) == '*') {
				int k = j + 1;
				if (k < p.length()) {
					while (i < s.length() && s.charAt(i) != p.charAt(k)) {
						i++;
					}
				} else {
					return true;
				}
				i++;
				j = k + 1;
			} else {
				return false;
			}
		}
		return i == s.length() && (j == p.length() || allstars(p.substring(j)));
	}

	public boolean allstars(String s) {
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) != '*')
				return false;
		}
		return true;
	}

	public boolean matching2(String s, String p) {
		if (p.length() == 0)
			return s.length() == 0;
		if (p.charAt(0) != '*') {
			if (s.charAt(0) == p.charAt(0))
				return matching2(s.substring(1), p.substring(1));
			return false;
		}
		int i = 1;
		while (i < s.length()) {
			if (matching2(s.substring(i), p.substring(1)))
				return true;
			i++;
		}
		return matching2(s, p.substring(1));
	}

	/*
	 * 排序好数组里面的出现次数超过1次的数字的最后一个index返回，让用binary search写了一下
	 */

	public int getLastIndex(int[] A, int target) {
		if (A.length == 0)
			return -1;
		int beg = 0, end = A.length - 1;
		int index = -1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A[mid] == target) {
				index = mid;
				break;
			} else if (A[mid] > target)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		if (index == -1)
			return -1;
		beg = index + 1;
		end = A.length - 1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A[mid] > target)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		return end;
	}

	/*
	 * Google interview There are n coins in a line. (Assume n is even). Two
	 * players take turns to take a coin from one of the ends of the line until
	 * there are no more coins left. The player with the larger amount of money
	 * wins. Would you rather go first or second? Does it matter? Assume that
	 * you go first, describe an algorithm to compute the maximum amount of
	 * money you can win.
	 */
	/*
	 * sol: Let d(i, j) be the maximum amount of money that you can win given
	 * the coins [i ... j] If we take A[i], then the max amount that the
	 * opponent will get is d(i + 1, j); If we take A[j], then the max amount
	 * that the opponent will get is d(i, j - 1);
	 */
	/*
	 * If we take A[i], d(i, j) = A[i] + min{d(i + 2, j), d(i - 1, j - 1)}; If
	 * we take A[j], d(i, j) = A[j] + min{d(i + 1, j - 1), d(i, j - 2)}. Hence,
	 * d(i, j) = max{A[i] + min{d(i + 2, j), d(i - 1, j - 1)}, A[j] + min{d(i +
	 * 1, j - 1), d(i, j - 2)}}, i < jd(i, j) = A[i], i == jd(i, j) = max{A[i],
	 * A[j]}, i == j - 1
	 */

	public int maxMoney(int[] coins) {
		int n = coins.length;
		int[][] dp = new int[n][n];

		for (int gap = 0; gap < n; gap++) {
			for (int i = 0; i < n - gap; i++) {
				int j = i + gap;
				if (i == j)
					dp[i][j] = coins[i];
				else if (i + 1 == j)
					dp[i][j] = Math.max(coins[i], coins[j]);
				else {
					dp[i][j] = Math
							.max(coins[i]
									+ Math.min(dp[i + 1][j - 1], dp[i + 2][j]),
									coins[j]
											+ Math.min(dp[i + 1][j - 1],
													dp[i][j - 2]));
				}
			}
		}
		return dp[0][n - 1];
	}

	// print the path: how the choose the coin
	public String maxMoneyFollowup(int[] coins) {
		int n = coins.length;
		int[][] dp = new int[n][n];

		for (int gap = 0; gap < n; gap++) {
			for (int i = 0; i < n - gap; i++) {
				int j = i + gap;
				if (i == j)
					dp[i][j] = coins[i];
				else if (i + 1 == j)
					dp[i][j] = Math.max(coins[i], coins[j]);
				else {
					dp[i][j] = Math
							.max(coins[i]
									+ Math.min(dp[i + 1][j - 1], dp[i + 2][j]),
									coins[j]
											+ Math.min(dp[i + 1][j - 1],
													dp[i][j - 2]));
				}
			}
		}
		StringBuilder sb = new StringBuilder();
		boolean myTurn = true;
		int beg = 0, end = n - 1;
		while (beg < end) {
			if (myTurn)
				sb.append("I take ");
			else
				sb.append("You take ");
			if (dp[beg + 1][end] < dp[beg][end - 1])
				sb.append(coins[beg++]);
			else
				sb.append(coins[end--]);
			myTurn = !myTurn;
			sb.append("\n");
		}
		sb.append("Total is " + dp[0][n - 1] + "\n");
		return sb.toString();
	}

	// airbnb
	/*
	 * Given a list of word and a target word, output all the words for each the
	 * edit distance with the target no greater than k. e.g. [abc, abd, abcd,
	 * adc], target "ac", k = 1, output = [abc, adc]
	 */
	public List<String> getKEditDistance(String[] words, String target, int k) {
		List<String> res = new ArrayList<String>();
		Trie trie = new Trie();
		for (String word : words) {
			trie.insert(word);
		}
		TrieNode root = trie.root;
		int[] preDist = new int[target.length() + 1];
		for (int i = 0; i < preDist.length; i++) {
			preDist[i] = i;
		}
		getKEditDistance("", target, root, k, preDist, res);
		return res;
	}

	public void getKEditDistance(String cur, String target, TrieNode root,
			int k, int[] preDist, List<String> res) {
		if (root.isLeaf) {
			if (preDist[target.length()] <= k) {
				res.add(cur);
			}
		}

		for (int i = 0; i < 26; i++) {
			if (root.children[i] == null)
				continue;
			int[] curDist = new int[target.length() + 1];
			curDist[0] = cur.length() + 1;
			for (int j = 1; j <= target.length(); j++) {
				if (target.charAt(j - 1) == (char) (i + 'a'))
					curDist[j] = preDist[j - 1];
				else
					curDist[j] = Math.min(Math.min(preDist[j - 1], preDist[j]),
							curDist[j - 1]) + 1;
			}
			getKEditDistance(cur + (char) (i + 'a'), target, root.children[i],
					k, curDist, res);
		}
	}

	/*
	 * 在一个整数数组中（有重复元素）找出有多少对，满足条件：他们的差等于k。 例如[1,5,5,2,4,6,7]，k=3，满足条件的对子有[1,4],
	 * [2,5], [2,5]（注意有两个5）,[4,7]，所以程序返回4。 这题比较tricky的地方在于k=0的情况需要另外考虑一下。
	 */

	public List<List<Integer>> findTwoMinusPairs(int[] A, int k) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (A.length < 2)
			return res;
		Arrays.sort(A);

		int i = 0, j = 1;
		while (i < A.length && j < A.length) {
			System.out.println(i + " " + j);
			if (A[j] - A[i] == k) {
				List<Integer> lst = new ArrayList<Integer>();
				lst.add(A[i]);
				lst.add(A[j]);
				int count1 = 1;
				int t = i + 1;
				while (t < A.length && A[i] == A[t++]) {
					count1++;
				}
				t = j + 1;
				int count2 = 1;
				while (t < A.length && A[j] == A[t++]) {
					count2++;
				}
				for (int count = 0; count < count1 * count2; count++) {
					res.add(lst);
				}
				i += count1;
				j += count2;

			} else if (A[j] - A[i] < k)
				j++;
			else
				i++;
		}
		return res;
	}

	/*
	 * 写一个函数float sumPossibility(int dice, int
	 * target)，就是投dice个骰子，求最后和为target的概率。 因为总共的可能性是6^dice，所以其实就是combination
	 * sum，求dice个骰子有多少种组合，使其和为target。 先用brute
	 * force的dfs来一个O(6^dice)指数复杂度的，然后要求优化，用dp
	 */

	public double sumPossibiluty(int dice, int target) {
		if (dice <= 0 || target <= 0)
			return 0;
		int sum = (int) Math.pow(6, dice);

		int[][] dp = new int[dice + 1][target + 1];
		dp[0][0] = 1;

		for (int i = 1; i <= dice; i++) {
			for (int j = 1; j <= target; j++) {
				for (int k = 1; k <= 6; k++) {
					if (j >= k) {
						dp[i][j] += dp[i - 1][j - k];
					}
				}
			}
		}
		return (double) dp[dice][target] / sum;
	}

	/*
	 * 给你一个二维矩阵，返回一个矩阵根据左右对称反转的矩阵。这个矩阵由3个量表示：width，height，byte［］。 例子： 这样一个二维矩阵
	 * ［［1，2，3］， ［4，5，6］］.
	 * 
	 * 
	 * 表示为： width ＝ 3， height ＝ 2， byte［］ ＝ ｛1，2，3，4，5，6｝； 然后求翻转后的一维矩阵。 得到res［］
	 * ＝ ｛3，2，1，6，5，4｝
	 */

	public int[] reverseMatrix(int width, int height, int[] nums) {
		int n = nums.length;
		// reverse(nums, 0, n-1);
		for (int i = 0, start = 0; i < height; i++, start += width) {
			reverse(nums, start, start + width - 1);
		}
		return nums;
	}

	public void reverse(int[] nums, int beg, int end) {
		while (beg < end) {
			int t = nums[beg];
			nums[beg] = nums[end];
			nums[end] = t;
			beg++;
			end--;
		}
	}

	public List<List<Integer>> palindromePairs(String[] words) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		Map<String, Integer> map = new HashMap<String, Integer>();

		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			map.put(reverseWord(word), i);
		}

		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			for (int j = 0; j <= word.length(); j++) {
				String sub = word.substring(0, j);
				String left = word.substring(j);
				if (map.containsKey(sub) && isPalindrome(left)) {
					System.out.println(map.get(sub) + ", " + i);
					List<Integer> lst = new ArrayList<Integer>();
					lst.add(i);
					lst.add(map.get(sub));
					if (!res.contains(lst) && i != map.get(sub))
						res.add(lst);
				}
			}
			for (int j = word.length(); j >= 0; j--) {
				String sub = word.substring(j);
				String left = word.substring(0, j);
				System.out.println("2 round " + sub + ", " + left);
				if (map.containsKey(sub) && isPalindrome(left)) {
					List<Integer> lst = new ArrayList<Integer>();
					lst.add(map.get(sub));
					lst.add(i);
					if (!res.contains(lst) && i != map.get(sub))
						res.add(lst);
				}
			}
		}
		return res;
	}

	public String reverseWord(String word) {
		StringBuilder sb = new StringBuilder(word);
		return sb.reverse().toString();
	}

	/*
	 * Uber Growth interview given_string = "iamhappy"; //=> [ "i", "am",
	 * "happy"]. // def _is_word_in_dictionary(given_string): // return
	 * given_string in ("i", "a", "am", "happy", "hello", "after", "noon",
	 * "afternoon") // # input: "iamhappy" // # output: ["i", "am", "happy”]
	 */

	public List<List<String>> wordInDicionary(String s, Set<String> dict) {
		List<List<String>> res = new ArrayList<List<String>>();
		List<String> sol = new ArrayList<String>();
		wordInDictionaryUtil(s, dict, sol, res);
		return res;
	}

	public void wordInDictionaryUtil(String s, Set<String> dict,
			List<String> sol, List<List<String>> res) {
		if (s.length() == 0) {
			res.add(new ArrayList<String>(sol));
		}

		for (int i = 1; i <= s.length(); i++) {
			String sub = s.substring(0, i);
			if (dict.contains(sub)) {
				sol.add(sub);
				wordInDictionaryUtil(s.substring(i), dict, sol, res);
				sol.remove(sol.size() - 1);
			}
		}
	}

	/*
	 * 0 1 2 3 4 5 6 7 8
	 * 
	 * have to have at least 4 pins no repeats can only connect unobstructed
	 * pins
	 * 
	 * 4025 <- not valid, 1 blocks 0 and 2 14025 <- ok, 1 already visited, 0 and
	 * 2 are not blocked
	 * 
	 * 
	 * input: string (ex: 0125) output: boolean (true if valid)
	 */

	public boolean validPin(int[][] matrix, String pin) {
		int m = matrix.length, n = matrix[0].length;
		boolean[][] visited = new boolean[m][n];

		Queue<Integer> que = new LinkedList<Integer>();
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (matrix[i][j] == pin.charAt(0) - '0') {
					return dfs(matrix, pin, visited, i, j, 0);
					// que.add(i*matrix[0].length+j);
					// break;
				}
			}
		}
		// int cur=0;
		// while(!que.isEmpty()){
		// int top=que.poll();
		// int row=top/n;
		// int col=top%n;
		// if(++cur==pin.length())
		// return true;
		// visited[row][col]=true;
		// if(row+1<m&&!visited[row+1][col]&&matrix[row+1][col]==pin.charAt(cur)){
		// que.add((row+1)*n+col);
		// }else if()
		// }

		return false;
	}

	public boolean dfs(int[][] matrix, String pin, boolean[][] visited, int i,
			int j, int cur) {
		if (cur == pin.length())
			return true;
		if (i < 0 || j < 0 || i >= matrix.length || j >= matrix[0].length
				|| matrix[i][j] != pin.charAt(cur) - '0') {
			return false;
		}

		visited[i][j] = true;
		boolean res = (i + 1 < matrix.length ? dfs(matrix, pin, visited,
				visited[i + 1][j] ? i + 2 : i + 1, j, cur + 1) : false)
				|| (i - 1 >= 0 ? dfs(matrix, pin, visited,
						visited[i - 1][j] ? i - 2 : i - 1, j, cur + 1) : false)
				|| (j - 1 >= 0 ? dfs(matrix, pin, visited, i,
						visited[i][j - 1] ? j - 2 : j - 1, cur + 1) : false)
				|| (j + 1 < matrix[0].length ? dfs(matrix, pin, visited, i,
						visited[i][j + 1] ? j + 2 : j + 1, cur + 1) : false)
				|| (i + 1 < matrix.length && j + 1 < matrix[0].length ? dfs(
						matrix, pin, visited, visited[i + 1][j + 1] ? i + 2
								: i + 1, visited[i + 1][j + 1] ? j + 2 : j + 1,
						cur + 1) : false)
				|| (i + 1 < matrix.length && j - 1 >= 0 ? dfs(matrix, pin,
						visited, visited[i + 1][j - 1] ? i + 2 : i + 1,
						visited[i + 1][j - 1] ? j - 2 : j - 1, cur + 1) : false)
				|| (i - 1 >= 0 && j - 1 >= 0 ? dfs(matrix, pin, visited,
						visited[i - 1][j - 1] ? i - 2 : i - 1,
						visited[i - 1][j - 1] ? j - 2 : j - 1, cur + 1) : false)
				|| (i - 1 >= 0 && j + 1 < matrix[0].length ? dfs(matrix, pin,
						visited, visited[i - 1][j + 1] ? i - 2 : i - 1,
						visited[i - 1][j + 1] ? j + 2 : j + 1, cur + 1) : false);

		visited[i][j] = false;
		return res;
	}

	// //Count number of ways to fill a “2 x n″ grid using “1 x 2″ tiles
	// tiles can be placed horizontally or vertically
	public int countWaysOfTiling(int n) {
		int[] dp = new int[n + 1];
		for (int i = 0; i <= n; i++) {
			if (i < 3)
				dp[i] = i;
			else
				dp[i] = dp[i - 1] + dp[i - 2];
		}

		return dp[n];
	}

	// Count number of ways to fill a “n x 4″ grid using “1 x 4″ tiles

	public int countWaysOfTilingFollowUp(int n) {
		int[] dp = new int[n + 1];
		for (int i = 0; i <= n; i++) {
			if (i < 4)
				dp[i] = 1;
			else if (i == 4)
				dp[i] = 2;
			else
				dp[i] = dp[i - 1] + dp[i - 4];
			// dp(i-1) : Place first tile horizontally
			// dp(n-4) : Place first tile vertically
			// which means 3 more tiles have
			// to be placed vertically.
		}
		return dp[n];
	}

	// Find Recurring Sequence in a Fraction
	public String fractionToDecimal2(int numr, int denr) {
		StringBuilder sb = new StringBuilder();
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		int res = numr / denr;
		sb.append(res + ".");
		int rem = numr % denr;
		while (rem != 0 && !map.containsKey(rem)) {
			map.put(rem, sb.length());
			rem *= 10;
			sb.append(rem / denr);
			rem %= denr;
		}
		return sb.toString();
	}

	public List<List<Integer>> splitLotteryNumbers(String s) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (s.length() < 7 || s.length() > 14)
			return res;
		List<Integer> sol = new ArrayList<Integer>();
		splitLotteryNumbersUtil(0, s, sol, res);
		return res;
	}

	public void splitLotteryNumbersUtil(int dep, String s, List<Integer> sol,
			List<List<Integer>> res) {
		if (dep == 6 && isValidLotteryNum(s)) {
			List<Integer> out = new ArrayList<Integer>(sol);
			out.add(Integer.parseInt(s));
			res.add(out);
		}

		for (int i = 0; i < s.length() && i < 2; i++) {
			String sub = s.substring(0, i + 1);
			if (isValidLotteryNum(sub)) {
				sol.add(Integer.parseInt(sub));
				splitLotteryNumbersUtil(dep + 1, s.substring(i + 1), sol, res);
				sol.remove(sol.size() - 1);
			}
		}
	}

	public boolean isValidLotteryNum(String s) {
		if (s.length() == 0)
			return false;
		int num = Integer.parseInt(s);
		return num >= 1 && num <= 59;
	}

	// Bin Packing Problem (Minimize number of used Bins)
	public int nextFit(int[] weight, int c) {
		int res = 1;
		int leftCap = c;
		for (int i = 0; i < weight.length; i++) {
			if (weight[i] > leftCap) {
				res++;
				leftCap = c - weight[i];
			} else
				leftCap -= weight[i];
		}
		return res;
	}

	public int firstFit(int[] weight, int c) {
		int n = weight.length;
		int[] bin_rem = new int[n]; // store remaining space in bins, there can
									// be at most n bins
		int res = 0;

		for (int i = 0; i < n; i++) {
			int j;
			for (j = 0; j < res; j++) {
				if (bin_rem[j] >= weight[i]) {
					bin_rem[j] -= weight[i];
					break;
				}
			}
			if (j == res) {
				bin_rem[res] = c - weight[i];
				res++;
			}
		}
		return res;
	}

	public int bestFit(int[] weight, int c) {
		int res = 0;
		int n = weight.length;
		int[] bin_rem = new int[n];

		for (int i = 0; i < n; i++) {
			int j;
			int min = c + 1, bi = 0;
			for (j = 0; j < res; j++) {
				if (bin_rem[j] >= weight[i] && bin_rem[j] - weight[i] < min) {
					min = bin_rem[j] - weight[i];
					bi = j;
				}
			}

			if (min == c + 1) {
				bin_rem[res] = c - weight[i];
				res++;
			} else {
				bin_rem[bi] -= weight[i];
			}
		}
		return res;
	}

	/*
	 * Suppose you have a long flowerbed in which some of the plots are planted
	 * and some are not. However, flowers cannot be planted in adjacent plots -
	 * they would compete for water and both would die. Given a flowerbed
	 * (represented as an array containing booleans), return if a given number
	 * of new flowers can be planted in it without violating the
	 * no-adjacent-flowers rule.
	 */
	public boolean canPlaceFlowers(List<Boolean> flowerbed, int numberToPlace) {
		if (numberToPlace == 0)
			return true;
		if (flowerbed.size() == 0)
			return false;
		if (flowerbed.size() == 1)
			return !flowerbed.get(0) && numberToPlace <= 1;
		// if (flowerbed.size() == 2)
		// return numberToPlace < 2
		// && (!flowerbed.get(0) && !flowerbed.get(1));
		int count = 0;
		int n = flowerbed.size();
		for (int i = 0; i < n; i++) {
			if (!flowerbed.get(i)) {
				if ((i == 0 && !flowerbed.get(i + 1))
						|| (i == n - 1 && !flowerbed.get(i - 1))
						|| (!flowerbed.get(i + 1) && !flowerbed.get(i - 1))) {
					flowerbed.set(i, true);
					count++;
					if (count == numberToPlace)
						return true;
				}
			}
		}
		return false;
	}

	/*
	 * Google Interview 1. Given two number a and b, find all the binary
	 * representations with length a and has b number of 1's and output in
	 * decimal in sorted order Example: input: a = 3, b = 1 [001, 010, 100]
	 * output: [1, 2, 4]
	 */

	public List<Integer> findNumbersWithNBits(int a, int b) {
		// int num=0;
		// for(int i=0;i<b;i++){
		// num|=1<<i;
		// }
		// String s=Integer.toBinaryString(num);
		// for(int i=0;i<a-b;i++){
		// s="0"+s;
		// }
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < a - b; i++) {
			sb.append("0");
		}
		for (int i = 0; i < b; i++) {
			sb.append("1");
		}
		List<Integer> res = new ArrayList<Integer>();
		boolean[] used = new boolean[sb.length()];
		findNumbersWithNBits(sb, "", used, res);
		return res;
	}

	public void findNumbersWithNBits(StringBuilder s, String sol,
			boolean[] used, List<Integer> res) {
		if (sol.length() == s.length()) {
			int num = Integer.parseInt(sol, 2);
			res.add(num);
		}
		for (int i = 0; i < s.length(); i++) {
			if (!used[i]) {
				if (i != 0 && s.charAt(i) == s.charAt(i - 1) && !used[i - 1])
					continue;
				used[i] = true;
				findNumbersWithNBits(s, sol + s.charAt(i), used, res);
				used[i] = false;
			}

		}
	}

	/*
	 * 给一个二维boolean array， true代表greyed， 要找出所有可能的正方形。比如：
	 * 
	 * 0 1 0 0 0 0 1 0 0
	 * 
	 * 一共有8个正方形（边长为1的7个，为2的1个，为3的0个）。注意matrix的边长可能不等
	 */
	public int countSquares(boolean[][] matrix) {
		int m = matrix.length;
		if (m == 0)
			return 0;
		int n = matrix[0].length;
		int[][] dp = new int[m][n];

		int count = 0;
		for (int i = 0; i < m; i++) {
			if (!matrix[i][0]) {
				dp[i][0] = 1;
				count++;
			}
		}
		for (int j = 1; j < n; j++) {
			if (!matrix[0][j]) {
				dp[0][j] = 1;
				count++;
			}
		}

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				if (!matrix[i][j]) {
					dp[i][j] = Math.min(dp[i - 1][j - 1],
							Math.min(dp[i - 1][j], dp[i][j - 1])) + 1;
				}
			}
		}
		for (int i = 0; i < m; i++) {
			System.out.println(Arrays.toString(dp[i]));
		}

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				count += dp[i][j];
			}
		}
		return count;
	}

	/*
	 * 1->2->3->4->5 ｜ 6->7->8. | 9 要flatten变成126978345
	 */

	public ListNodeWithDown flattenList(ListNodeWithDown head) {
		if (head == null)
			return null;
		Stack<ListNodeWithDown> stk = new Stack<ListNodeWithDown>();
		ListNodeWithDown cur = head;
		ListNodeWithDown dummy = new ListNodeWithDown(0);
		ListNodeWithDown pre = dummy;
		while (cur != null) {
			if (cur.down == null) {
				pre.next = cur;
				cur = cur.next;
				pre = pre.next;
			} else {
				stk.push(cur.next);
				pre.next = cur;
				cur = cur.down;
				pre = pre.next;
			}
			if (cur == null && !stk.isEmpty()) {
				cur = stk.pop();
			}
		}
		return dummy.next;

	}

	// constance space
	public ListNodeWithDown flattenListRecursive(ListNodeWithDown head) {
		if (head == null)
			return head;
		if (head.next != null && head.next != null) {
			ListNodeWithDown pnext = head.next;
			head.next = flattenListRecursive(head.down);
			ListNodeWithDown cur = head;
			while (cur.next != null) {
				cur = cur.next;
			}
			cur.next = flattenListRecursive(pnext);
		} else if (head.down == null) {
			head.next = flattenListRecursive(head.next);
		} else {
			head.next = flattenListRecursive(head.down);
		}
		return head;
	}

	/*
	 * Given a linked list where every node represents a linked list and
	 * contains two pointers of its type: (i) Pointer to next node in the main
	 * list (we call it ‘right’ pointer in below code) (ii) Pointer to a linked
	 * list where this node is head (we call it ‘down’ pointer in below code).
	 * All linked lists are sorted. See the following example
	 * 
	 * 5 -> 10 -> 19 -> 28 | | | | V V V V 7 20 22 35 | | | V V V 8 50 40 | | V
	 * V 30 45 Write a function flatten() to flatten the lists into a single
	 * linked list. The flattened linked list should also be sorted. For
	 * example, for the above input list, output list should be
	 * 5->7->8->10->19->20->22->28->30->35->40->45->50.
	 */

	public ListNodeWithDown flatten(ListNodeWithDown head) {
		if (head == null || head.next == null)
			return head;
		return merge(head, flatten(head.next));
	}

	public ListNodeWithDown merge(ListNodeWithDown head1, ListNodeWithDown head2) {
		if (head1 == null || head2 == null)
			return head1 == null ? head2 : head1;
		ListNodeWithDown res = null;
		if (head1.val < head2.val) {
			res = head1;
			res.down = merge(head1.down, head2);
		} else {
			res = head2;
			res.down = merge(head1, head2.down);
		}
		return res;
	}

	// flatten multi-level list, parallel
	public ListNodeWithDown flattenList2(ListNodeWithDown node) {
		if (node == null)
			return null;
		ListNodeWithDown tail = node;
		while (tail.next != null) {
			tail = tail.next;
		}

		ListNodeWithDown cur = node;
		while (cur != tail) {
			if (cur.down != null) {
				tail.next = cur.down;
				ListNodeWithDown t = cur.down;
				while (t.next != null) {
					t = t.next;
				}
				tail = t;
			}
			cur = cur.next;
		}
		return node;
	}

	/*
	 * First Number : 1007 Second Number : 93 Addition : 1100
	 */
	public ListNode LinkedListAddtionForwardOrder(ListNode head1, ListNode head2) {
		if (head1 == null || head2 == null)
			return head1 == null ? head2 : head1;
		int len1 = getLength(head1);
		int len2 = getLength(head2);
		if (len1 < len2) {
			ListNode t = head1;
			head1 = head2;
			head2 = t;
		}
		int diff = len1 - len2;
		while (diff > 0) {
			ListNode node = new ListNode(0);
			node.next = head2;
			head2 = node;
			diff--;
		}

		ListNode newHead = addTwoListsRecursion(head1, head2);
		if (carry == 1) {
			ListNode n = new ListNode(1);
			n.next = newHead;
			newHead = n;
		}
		return newHead;
	}

	// add two lists with same length
	// carry
	int carry = 0;
	ListNode newHead = null;

	public ListNode addTwoListsRecursion(ListNode head1, ListNode head2) {
		if (head1 == null && head2 == null)
			return null;
		addTwoListsRecursion(head1.next, head2.next);
		int sum = head1.val + head2.val + carry;
		carry = sum / 10;
		sum %= 10;

		ListNode node = new ListNode(sum);
		if (newHead == null)
			newHead = node;
		else {
			node.next = newHead;
			newHead = node;
		}
		return newHead;
	}

	public int getLength(ListNode head) {
		if (head == null)
			return 0;
		int len = 0;
		ListNode cur = head;
		while (cur != null) {
			len++;
			cur = cur.next;
		}
		return len;
	}

	// Print elements of a matrix in diagonal order
	public List<List<Integer>> printDiagonalElements(int[][] matrix) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		int m = matrix.length;
		if (m == 0)
			return res;
		int n = matrix[0].length;
		for (int i = 0; i < m + n - 1; i++) {
			int row, col;
			if (i < m) {
				row = i;
				col = 0;
			} else {
				row = m - 1;
				col = (i + 1) % m;
			}
			List<Integer> diagonal = new ArrayList<Integer>();
			while (row >= 0 && col < n) {
				diagonal.add(matrix[row][col]);
				row--;
				col++;
			}
			res.add(diagonal);
		}
		return res;
	}

	public List<List<Integer>> printDiagonalElements2(int[][] matrix) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		int m = matrix.length;
		if (m == 0)
			return res;
		int n = matrix[0].length;
		for (int i = 0; i < m; i++) {
			List<Integer> diagonal = new ArrayList<Integer>();
			int row = i, col = 0;
			while (row >= 0 && col < n) {
				diagonal.add(matrix[row][col]);
				row--;
				col++;
			}
			res.add(diagonal);
		}

		for (int i = 1; i < n; i++) {
			List<Integer> diagonal = new ArrayList<Integer>();
			int row = m - 1, col = i;
			while (row >= 0 && col < n) {
				diagonal.add(matrix[row][col]);
				row--;
				col++;
			}
			res.add(diagonal);
		}
		return res;
	}

	// 输出连续递增 （必须每一个数字比前一个大1）subsequence的最大长度， 比如
	// 1,2,3,4,5,6,10,11,12,100,200答案是 6 （1,2,3,4,5,6）
	public List<Integer> longestSubsequence(int[] A) {
		int n = A.length;
		int[] dp = new int[n];
		Arrays.fill(dp, 1);

		int max = 1;
		int index = -1;
		for (int i = 1; i < n; i++) {
			if (A[i] == A[i - 1] + 1) {
				dp[i] = dp[i - 1] + 1;
				if (dp[i] > max) {
					max = dp[i];
					index = i;
				}
			}
		}
		List<Integer> res = new ArrayList<Integer>();
		System.out.println(index + " " + max);
		for (int i = index - max + 1; i <= index; i++) {
			res.add(A[i]);
		}
		return res;
	}

	/*
	 * max-flow Ford-Fulkerson Algorithm for Maximum Flow Problem
	 */
	public int fordFulkerson(int graph[][], int s, int t) {
		int V = graph.length;

		// Residual graph where rGraph[i][j] indicates
		// residual capacity of edge from i to j (if there
		// is an edge. If rGraph[i][j] is 0, then there is
		// not)
		int[][] rGraph = new int[V][V];
		for (int u = 0; u < V; u++) {
			for (int v = 0; v < V; v++) {
				rGraph[u][v] = graph[u][v];
			}
		}
		int[] parent = new int[V];
		int maxFlow = 0;
		while (bfsAugment(rGraph, s, t, parent)) {
			// Find minimum residual capacity of the edhes
			// along the path filled by BFS.
			int pathFlow = Integer.MAX_VALUE;
			for (int v = t; v != s; v = parent[v]) {
				int u = parent[v];
				pathFlow = Math.min(pathFlow, rGraph[u][v]);
			}

			// update residual capacities of the edges and
			// reverse edges along the path
			for (int v = t; v != s; v = parent[v]) {
				int u = parent[v];
				rGraph[u][v] -= pathFlow;
				rGraph[v][u] += pathFlow;
			}
			maxFlow += pathFlow;
		}
		return maxFlow;
	}

	public boolean bfsAugment(int[][] rGraph, int s, int t, int[] parent) {
		int V = rGraph.length;
		boolean[] visited = new boolean[V];
		Queue<Integer> que = new LinkedList<Integer>();
		que.add(s);
		parent[s] = -1;
		visited[s] = true;
		while (!que.isEmpty()) {
			int u = que.poll();
			for (int v = 0; v < V; v++) {
				if (!visited[v] && rGraph[u][v] > 0) {
					que.offer(v);
					visited[v] = true;
					parent[v] = u;
				}
			}
		}
		return visited[t];
	}

	// A DFS based function to find all reachable vertices from s. The function
	// marks visited[i] as true if i is reachable from s. The initial values in
	// visited[] must be false. We can also use BFS to find reachable vertices
	public void dfs(int[][] rGraph, int s, boolean visited[]) {
		int V = rGraph.length;
		visited[s] = true;
		for (int i = 0; i < V; i++)
			if (rGraph[s][i] > 0 && !visited[i])
				dfs(rGraph, i, visited);
	}

	public void minCut(int[][] graph, int s, int t) {
		int V = graph.length;
		int[][] rGraph = new int[V][V];
		for (int u = 0; u < V; u++) {
			for (int v = 0; v < V; v++) {
				rGraph[u][v] = graph[u][v];
			}
		}

		int[] parent = new int[V];
		int maxFlow = 0;
		while (bfsAugment(rGraph, s, t, parent)) {
			int pathFlow = Integer.MAX_VALUE;
			for (int v = t; v != s; v = parent[v]) {
				int u = parent[v];
				pathFlow = Math.min(pathFlow, rGraph[u][v]);
			}

			for (int v = t; v != s; v = parent[v]) {
				int u = parent[v];
				rGraph[u][v] -= pathFlow;
				rGraph[v][u] += pathFlow;
			}
			maxFlow += pathFlow;
		}
		// Flow is maximum now, find vertices reachable from s
		boolean[] visited = new boolean[V];
		dfs(rGraph, s, visited);

		for (int u = 0; u < V; u++) {
			for (int v = 0; v < V; v++) {
				if (visited[u] && !visited[v] && graph[u][v] > 0) {
					System.out.println("edge is: " + u + " -- " + v);
				}
			}
		}
	}

	// minimum spanning tree
	public void primMST(int[][] graph) {
		int V = graph.length;
		// array to store constructed MST
		int[] parent = new int[V];
		// key values used to pick min weight edge
		int[] key = new int[V];
		// To represent set of vertices not yet included in MST
		boolean[] mstSet = new boolean[V];

		for (int i = 0; i < V; i++) {
			key[i] = Integer.MAX_VALUE;
			mstSet[i] = false;
		}
		// always include the first vertex
		// Make key 0 so that this vertex is
		// picked as first vertex
		key[0] = 0;
		// first node is the root of MST
		parent[0] = -1;

		for (int count = 0; count < V; count++) {
			int u = minKey(key, mstSet);
			mstSet[u] = true;

			// Update key value and parent index of the adjacent
			// vertices of the picked vertex. Consider only those
			// vertices which are not yet included in MST
			for (int v = 0; v < V; v++) {
				if (graph[u][v] != 0 && !mstSet[v] && graph[u][v] < key[v]) {
					key[v] = graph[u][v];
					parent[v] = u;
				}
			}
		}
		printMST(parent, graph);
	}

	public int minKey(int[] key, boolean[] mstSet) {
		int min = Integer.MAX_VALUE;
		int index = -1;

		for (int i = 0; i < key.length; i++) {
			if (!mstSet[i] && key[i] < min) {
				min = key[i];
				index = i;
			}
		}
		return index;
	}

	public void printMST(int[] parent, int[][] graph) {
		System.out.println("Edge    Weight");
		for (int i = 1; i < parent.length; i++) {
			System.out.println(parent[i] + " - " + i + "     "
					+ graph[parent[i]][i]);
		}
	}

	/*
	 * 一个字符串里面word和word之间有空格让我把word分开来存在一个List<String>里面返回。但是有特殊情况 就是比如“hello
	 * thank you”我就返回“hello”“thank”“you” 但是如果是“my name is \“Donald
	 * duck\””就应该把Donald duck认作一个词。 还有比如“ \“ \” ”就应该返回引号里面的那个词（就是那几个空格）
	 */

	// "My name is \" \" and \"Donald Trump\""
	public List<String> splitWords(String s) {
		List<String> res = new ArrayList<String>();
		StringBuilder sb = new StringBuilder();

		boolean quot = false;
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == '\"' && quot) {
				res.add(sb.toString());
				sb = new StringBuilder();
				quot = !quot;
			} else if (s.charAt(i) == '\"' && !quot) {
				quot = !quot;
			} else if (s.charAt(i) == ' ' && !quot && sb.length() > 0) {
				res.add(sb.toString());
				sb = new StringBuilder();
			} else
				sb.append(s.charAt(i));
		}
		if (sb.length() > 0) {
			res.add(sb.toString());
		}
		System.out.println(res);
		return res;
	}

	public void sinkZeros(TreeNode root) {
		if (root == null || root.left == null && root.right == null)
			return;
		if (root.val == 0) {
			pushDownZero(root);
		}
		sinkZeros(root.left);
		sinkZeros(root.right);

	}

	public void pushDownZero(TreeNode root) {
		if (root == null || root.left == null && root.right == null)
			return;
		if (root.left != null && root.left.val != 0) {
			int t = root.left.val;
			root.left.val = root.val;
			root.val = t;
			pushDownZero(root.left);
		}
		if (root.right != null && root.right.val != 0) {
			int t = root.right.val;
			root.right.val = root.val;
			root.val = t;
			pushDownZero(root.right);
		}
	}

	// ski
	public int getLongestPath(int[][] matrix) {
		int n = matrix.length;
		if (n == 0)
			return 0;
		int m = matrix[0].length;
		int max = 1;
		int[][] dp = new int[n][m];

		int r = -1, c = -1;
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				dfs(matrix, i, j, dp);
				if (dp[i][j] > max) {
					max = dp[i][j];
					r = i;
					c = j;
				}
			}
		}

		List<Integer> path = new ArrayList<Integer>();
		int t = max;
		while (t > 0) {
			path.add(matrix[r][c]);
			if (r + 1 < n && dp[r + 1][c] == dp[r][c] - 1) {
				r = r + 1;
			} else if (r - 1 >= 0 && dp[r - 1][c] == dp[r][c] - 1) {
				r = r - 1;
			} else if (c + 1 < m && dp[r][c + 1] == dp[r][c] - 1) {
				c = c + 1;
			} else if (c - 1 >= 0 && dp[r][c - 1] == dp[r][c] - 1) {
				c = c - 1;
			}
			t--;
		}
		System.out.println(path);

		return max;
	}

	public int dfs(int[][] matrix, int i, int j, int[][] dp) {
		if (dp[i][j] != 0)
			return dp[i][j];
		int len = 1;
		if (i + 1 < matrix.length && matrix[i + 1][j] < matrix[i][j]) {
			len = Math.max(len, dfs(matrix, i + 1, j, dp) + 1);
		}
		if (i - 1 >= 0 && matrix[i - 1][j] < matrix[i][j]) {
			len = Math.max(len, dfs(matrix, i - 1, j, dp) + 1);
		}
		if (j + 1 < matrix[0].length && matrix[i][j + 1] < matrix[i][j]) {
			len = Math.max(len, dfs(matrix, i, j + 1, dp) + 1);
		}
		if (j - 1 >= 0 && matrix[i][j - 1] < matrix[i][j]) {
			len = Math.max(len, dfs(matrix, i, j - 1, dp) + 1);
		}
		dp[i][j] = len;
		return len;
	}

	/*
	 * 给一个array,找出最长的连续片段 比如：[5,2,3,4,5,8,9] 就是2，3，4，5，返回4 O(N), space: O(1)
	 */

	public int longestConsecutiveSequence(int[] A) {
		if (A.length < 2)
			return A.length;
		int max = 1;
		int cur = 1;
		for (int i = 1; i < A.length; i++) {
			System.out.println(A[i] + " " + A[i - 1]);
			if (A[i] == A[i - 1] + 1) {
				cur++;
			} else {
				max = Math.max(max, cur);
				cur = 1;
			}
		}
		max = Math.max(cur, max);
		return max;
	}

	/*
	 * 给两个calender schedule (start time sorted), 要设定的meeting time interval，
	 * 返回最小的可设置schedule的起始时间。 sched A: [0,10], [10, 15], [13, 20] sched B:
	 * [0,5], [27, 33] meeting_time = 5, return 20
	 */

	public int findLatestStartTime(List<Interval> sched1,
			List<Interval> sched2, int meeting_time) {
		List<Interval> schedA = mergeInterval(sched1);
		List<Interval> schedB = mergeInterval(sched2);
		int len1 = schedA.size(), len2 = schedB.size();
		int i = 0, j = 0;
		while (i < len1 && j < len2) {
			Interval a = schedA.get(i);
			Interval b = schedB.get(j);
			System.out.println(a + ": " + b);
			if (a.end < b.start && b.start - a.end >= meeting_time)
				return a.end;
			else if (b.end < a.start && a.start - b.end >= meeting_time)
				return b.end;
			else {
				if (a.end <= b.start || a.end > b.start && a.end <= b.end)
					i++;
				else if (b.end <= a.start || b.end > a.start && b.end <= a.end)
					j++;
				else {
					i++;
					j++;
				}
			}
		}
		while (i < len1 - 1) {
			Interval a = schedA.get(i);
			Interval b = schedA.get(i + 1);
			if (b.start - a.end >= meeting_time)
				return a.end;
			else
				i++;
		}
		while (j < len2 - 1) {
			Interval a = schedB.get(j);
			Interval b = schedB.get(j + 1);
			if (b.start - a.end >= meeting_time)
				return a.end;
			else
				j++;
		}

		return Math.max(schedA.get(len1 - 1).end, schedB.get(len2 - 1).end);
	}

	public List<Interval> mergeInterval(List<Interval> intervals) {
		if (intervals.size() < 2)
			return intervals;
		List<Interval> res = new ArrayList<Interval>();
		res.add(intervals.get(0));
		for (int i = 1; i < intervals.size(); i++) {
			Interval interval = intervals.get(i);
			Interval last = res.get(res.size() - 1);
			if (interval.start > last.end)
				res.add(interval);
			else
				last.end = Math.max(last.end, interval.end);
		}
		return res;
	}

	public List<Interval> mergeIntervals(List<Interval> intervals1,
			List<Interval> intervals2) {
		List<Interval> res = new ArrayList<Interval>();
		int i = 0, j = 0;

		while (i < intervals1.size() && j < intervals2.size()) {
			Interval i1 = intervals1.get(i);
			Interval i2 = intervals2.get(j);
			if (res.size() == 0) {
				if (i1.start < i2.start) {
					res.add(i1);
					i++;
				} else {
					res.add(i2);
					j++;
				}
			} else {
				Interval last = res.get(res.size() - 1);
				if (i1.start < i2.start) {
					if (i1.start > last.end) {
						res.add(i1);
					} else {
						last.end = Math.max(last.end, i1.end);
					}
					i++;
				} else {
					if (i2.start > last.end) {
						res.add(i2);
					} else {
						last.end = Math.max(last.end, i2.end);
					}
					j++;
				}
			}
		}

		while (i < intervals1.size()) {
			Interval i1 = intervals1.get(i);
			Interval last = res.get(res.size() - 1);
			if (i1.start > last.end)
				res.add(i1);
			else
				last.end = Math.max(last.end, i1.end);
			i++;
		}

		while (j < intervals2.size()) {
			Interval i2 = intervals2.get(j);
			Interval last = res.get(res.size() - 1);
			if (i2.start > last.end)
				res.add(i2);
			else
				last.end = Math.max(last.end, i2.end);
			j++;
		}
		return res;
	}

	/*
	 * Suppose we have a method "getLongestSubstring(String s, int m)" which
	 * finds the longest substring with exactly M distinct characters.
	 * 
	 * Examples:
	 * 
	 * "ABACAAAB" M=2 -> "ACAAA"
	 */
	public String lengthOfLongestSubstringKDistinct(String s, int k) {
		if (s.length() < k)
			return "";
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		int start = 0;
		int maxLen = 0;
		int winStart = 0;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (!map.containsKey(c)) {
				if (map.size() < k)
					map.put(c, 1);
				else {
					if (i - start > maxLen) {
						maxLen = i - start;
						winStart = start;
					}
					while (map.size() == k) {
						char c1 = s.charAt(start);

						if (map.get(c1) == 1) {
							map.remove(c1);
						} else {
							map.put(c1, map.get(c1) - 1);
						}
						start++;
					}
					map.put(c, 1);
				}
			} else {
				map.put(c, map.get(c) + 1);
				if (i - start > maxLen) {
					maxLen = i - start;
					winStart = start;
				}
			}
		}
		System.out.println(s + ": " + winStart + "--" + maxLen);
		if (map.size() != k)
			return "";
		return s.substring(winStart, winStart + maxLen);
	}

	public boolean isPowerOfFour(int num) {
		return (num > 0) && ((num & (num - 1)) == 0)
				&& ((num & 0x55555555) == num);
	}

	public String longestCommonPrefix(String[] strs) {
		if (strs.length == 0)
			return "";
		String s = strs[0];
		for (int i = 0; i < s.length(); i++) {
			for (int j = 1; j < strs.length; j++) {
				if (s.charAt(i) != strs[j].charAt(i))
					return s.substring(0, i);
			}
		}
		return s;
	}

	class NumFreq {
		int num, freq;

		public NumFreq(int num, int freq) {
			this.num = num;
			this.freq = freq;
		}
	}

	public List<Integer> topKFrequent(int[] nums, int k) {
		List<Integer> res = new ArrayList<Integer>();
		if (nums.length < k)
			return res;
		PriorityQueue<NumFreq> heap = new PriorityQueue<NumFreq>(k,
				new Comparator<NumFreq>() {
					@Override
					public int compare(NumFreq nf1, NumFreq nf2) {
						return nf1.freq - nf2.freq;
					}
				});
		Map<Integer, Integer> count = new HashMap<Integer, Integer>();
		for (int num : nums) {
			if (!count.containsKey(num))
				count.put(num, 0);
			count.put(num, count.get(num) + 1);
		}

		for (Map.Entry<Integer, Integer> entry : count.entrySet()) {
			NumFreq nf = new NumFreq(entry.getKey(), entry.getValue());
			if (heap.size() < k)
				heap.offer(nf);
			else {
				if (heap.peek().freq < nf.freq) {
					heap.poll();
					heap.offer(nf);
				}
			}
		}

		while (!heap.isEmpty()) {
			res.add(heap.poll().num);
		}
		return res;
	}

	public ArrayList<ArrayList<String>> levelOrderInterval(TreeNode root) {
		ArrayList<ArrayList<String>> result = new ArrayList<ArrayList<String>>();
		if (root == null) {
			return result;
		}
		HashMap<TreeNode, Integer> nodeMap = new HashMap<TreeNode, Integer>();
		int h = getTreeHeight(root);
		int maxsize = (int) Math.pow(2, h);
		nodeMap.put(root, maxsize / 2);
		Queue<TreeNode> que = new LinkedList<TreeNode>();

		que.offer(root);

		while (!que.isEmpty()) {
			int size = que.size();
			ArrayList<String> curList = new ArrayList<String>();
			for (int k = 0; k < maxsize; k++) {
				curList.add(" ");
			}
			for (int i = 0; i < size; i++) {
				TreeNode curNode = que.poll();
				int pos = nodeMap.get(curNode);
				curList.set(pos, "" + curNode.val);
				if (curNode.left != null) {
					que.offer(curNode.left);
					nodeMap.put(curNode.left, pos / 2);
				}
				if (curNode.right != null) {
					que.offer(curNode.right);
					nodeMap.put(curNode.right, pos + (maxsize - pos) / 2);
				}
			}
			result.add(curList);
		}
		for (int i = 0; i < result.size(); i++) {
			System.out.println(result.get(i));
		}
		return result;
	}

	public int getTreeHeight(TreeNode root) {
		if (root == null)
			return 0;
		int left = getTreeHeight(root.left);
		int right = getTreeHeight(root.right);
		return left > right ? left + 1 : right + 1;
	}

	// Encodes a list of strings to a single string.
	public String encode(List<String> strs) {
		StringBuilder sb = new StringBuilder();
		for (String s : strs) {
			sb.append(s.length()).append('#').append(s);
		}
		return sb.toString();
	}

	// Decodes a single string to a list of strings.
	public List<String> decode(String s) {
		List<String> res = new ArrayList<String>();
		int i = 0;
		while (i < s.length()) {
			int sharp = s.indexOf("#", i);
			int len = Integer.valueOf(s.substring(i, sharp));
			res.add(s.substring(sharp + 1, sharp + 1 + len));
			i = sharp + 1 + len;
		}
		return res;
	}

	/*
	 * 给一个很长的数字字符串，要求对其做除法（返回类型为double）。由于数字很长，不能直接parse成数字，需要一点一点做大除法。
	 */

	public double largeNumDivide(String s1, String s2, int n) {
		long divisor = Long.parseLong(s2);
		long dividend = 0;
		int i = 0;
		StringBuilder sb = new StringBuilder();
		while (dividend < divisor) {
			if (i < s1.length()) {
				dividend = dividend * 10 + (s1.charAt(i) - '0');
				i++;
			} else
				break;
		}

		while (i < s1.length()) {
			sb.append(dividend / divisor);
			dividend %= divisor;
			dividend = dividend * 10 + (s1.charAt(i) - '0');
			i++;
		}
		System.out.println("cur: " + dividend);
		// while (i < s1.length()) {
		// if (dividend < divisor) {
		// dividend = dividend * 10 + (s1.charAt(i) - '0');
		// if(sb.length()!=0)
		// sb.append(0);
		// i++;
		// }else{
		// sb.append(dividend/divisor);
		// dividend%=divisor;
		// }
		// System.out.println("temp: "+dividend);
		// }

		sb.append(dividend / divisor).append(".");
		dividend %= divisor;
		if (dividend == 0)
			return Double.parseDouble(sb.toString());
		dividend *= 10;
		// if(dividend/divisor==0){
		// sb.append(".");
		// }else{
		// sb.append(dividend/divisor).append(".");
		// dividend%=divisor;
		// }
		while (dividend < divisor) {
			dividend *= 10;
			sb.append(0);
		}

		for (int k = 0; k <= n; k++) {
			sb.append(dividend / divisor);
			dividend %= divisor;
			dividend *= 10;
		}
		if (sb.charAt(sb.length() - 1) - '0' >= 5) {
			return Double.parseDouble(sb.substring(0, sb.length() - 2)
					+ (sb.charAt(sb.length() - 2) - '0' + 1));
		}
		return Double.parseDouble(sb.toString());
	}

	public int shortestDistance(int[][] grid) {
		int m = grid.length;
		if (m == 0)
			return 0;
		int res = Integer.MAX_VALUE;
		int n = grid[0].length;
		int[][] distance = new int[m][n];
		int[][] reach = new int[m][n];
		int buildingNum = 0;
		int[][] dirs = { { 1, 0 }, { -1, 0 }, { 0, 1 }, { 0, -1 } };
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (grid[i][j] == 1) {
					buildingNum++;
					boolean[][] visited = new boolean[m][n];
					int level = 1;
					Queue<Integer> que = new LinkedList<Integer>();
					que.offer(i * n + j);

					while (!que.isEmpty()) {
						int size = que.size();
						for (int q = 0; q < size; q++) {
							int first = que.poll();
							int row = first / n;
							int col = first % n;
							for (int k = 0; k < 4; k++) {
								int x = row + dirs[k][0];
								int y = col + dirs[k][1];
								if (x >= 0 && y >= 0 && x < m && y < n
										&& grid[x][y] == 0 && !visited[x][y]) {
									distance[x][y] += level;
									reach[x][y]++;
									visited[x][y] = true;
									que.offer(x * n + y);
								}
							}
						}
						level++;
					}
				}
			}
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (reach[i][j] == buildingNum && grid[i][j] == 0) {
					res = Math.min(res, distance[i][j]);
				}
			}
		}
		return res == Integer.MAX_VALUE ? -1 : res;
	}

	// 给个下面的route， 然后input：string source, string destination.
	// 打印出所有的可能路线。example, source=NY, dest=LA; return NY->LA, NY->DC->LA
	// , NY->Chicago->DC->LA.

	public List<String> findAllRoutes(String s, String d, String[][] tickets) {
		Map<String, List<String>> map = new HashMap<String, List<String>>();
		for (String[] ticket : tickets) {
			if (!map.containsKey(ticket[0]))
				map.put(ticket[0], new ArrayList<String>());
			map.get(ticket[0]).add(ticket[1]);
		}
		Set<String> visited = new HashSet<String>();
		List<String> res = new ArrayList<String>();
		dfsHelper(s, d, visited, map, "", res);
		return res;
	}

	public void dfsHelper(String s, String d, Set<String> visited,
			Map<String, List<String>> map, String route, List<String> res) {
		if (s.equals(d)) {
			res.add(route + s);
			return;
		}
		route += s + "->";
		List<String> destinations = map.get(s);
		for (String des : destinations) {
			if (!visited.contains(des)) {
				visited.add(s);
				dfsHelper(des, d, visited, map, route, res);
				visited.remove(s);
			}
		}
	}

	public int maxKilledEnemies(char[][] grid) {
		int m = grid.length;
		if (m == 0)
			return 0;
		int n = grid[0].length;
		int rowHit = 0;
		int[] colHit = new int[n];
		int res = 0;
		int col = -1, row = -1;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (j == 0 || grid[i][j - 1] == 'W') {
					rowHit = 0;
					for (int k = j; k < n && grid[i][k] != 'W'; k++) {
						rowHit += grid[i][k] == 'E' ? 1 : 0;
					}
				}

				if (i == 0 || grid[i - 1][j] == 'W') {
					colHit[j] = 0;
					for (int k = i; k < m && grid[k][j] != 'W'; k++) {
						colHit[j] += grid[k][j] == 'E' ? 1 : 0;
					}
				}

				if (grid[i][j] == '0') {
					if (rowHit + colHit[j] > res) {
						row = i;
						col = j;
					}
					res = Math.max(res, rowHit + colHit[j]);
				}
			}
		}
		System.out.println(row + ", " + col);
		return res;
	}

	// naive approach
	public int maxKilledEnemies2(char[][] grid) {
		int m = grid.length;
		if (m == 0)
			return 0;
		int n = grid[0].length;
		int res = 0;
		int row = -1, col = -1;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (grid[i][j] == '0') {
					if (searchEnemies(grid, i, j) > res) {
						row = i;
						col = j;
					}
					res = Math.max(res, searchEnemies(grid, i, j));
				}
			}
		}
		System.out.println(row + ", " + col);
		return res;
	}

	public int searchEnemies(char[][] grid, int row, int col) {
		int res = 0;
		for (int i = row; i < grid.length; i++) {
			if (grid[i][col] == 'W')
				break;
			res += grid[i][col] == 'E' ? 1 : 0;
		}

		for (int i = row; i >= 0; i--) {
			if (grid[i][col] == 'W')
				break;
			res += grid[i][col] == 'E' ? 1 : 0;
		}

		for (int i = col; i < grid[0].length; i++) {
			if (grid[row][i] == 'W')
				break;
			res += grid[row][i] == 'E' ? 1 : 0;
		}

		for (int i = col; i >= 0; i--) {
			if (grid[row][i] == 'W')
				break;
			res += grid[row][i] == 'E' ? 1 : 0;
		}

		return res;
	}

	public boolean wordBreakP(String s, Set<String> wordDict) {
		int n = s.length();
		boolean[] dp = new boolean[n + 1];
		dp[0] = true;
		for (int i = 1; i <= n; i++) {
			for (int j = 0; j < i; j++) {
				if (dp[j] && wordDict.contains(s.substring(j, i))) {
					dp[i] = true;
					break;
				}
			}
		}
		return dp[n];
	}

	public List<List<Integer>> findLeaves(TreeNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		findLeavesHelper(root, res);
		return res;

	}

	public int findLeavesHelper(TreeNode root, List<List<Integer>> res) {
		if (root == null)
			return -1;
		int height = 1 + Math.max(findLeavesHelper(root.left, res),
				findLeavesHelper(root.right, res));
		if (res.size() < height + 1) {
			res.add(new ArrayList<Integer>());
		}
		res.get(height).add(root.val);
		return height;
	}

	public int kthSmallest(int[][] matrix, int k) {
		int m = matrix.length;
		int n = matrix[0].length;
		PriorityQueue<Tuple> que = new PriorityQueue<Tuple>();
		for (int i = 0; i < m; i++) {
			que.offer(new Tuple(i, 0, matrix[i][0]));
		}

		for (int i = 0; i < k - 1; i++) {
			Tuple t = que.poll();
			if (t.c < n - 1)
				que.offer(new Tuple(t.r, t.c + 1, matrix[t.r][t.c + 1]));
		}
		return que.poll().v;
	}

	public int combinationSum4(int[] nums, int target) {
        if(nums.length==0||target<0)
            return 0;
        if(target==0)
            return 1;
        int count = 0;
        for(int num : nums){
            if(target>=num)
                count+=combinationSum4(nums, target-num);
        }
        return count;
    }
	
	public int numberOfPatterns(int m, int n) {
		int[][] jumps=new int[10][10];
		boolean[] visited=new boolean[10];
		jumps[1][3]=jumps[3][1]=2;
		jumps[4][6]=jumps[6][4]=5;
		jumps[7][9]=jumps[9][7]=8;
		jumps[1][7]=jumps[7][1]=4;
		jumps[2][8]=jumps[8][2]=5;
		jumps[3][9]=jumps[9][3]=6;
		jumps[1][9]=jumps[9][1]=jumps[3][7]=jumps[7][3]=5;
		
		int res = 0;
		res+=numberOfPatternsUtil(1, 1, 0, m, n, jumps, visited)*4;// 1,3,7, 9 symmetric
		res+=numberOfPatternsUtil(2, 1, 0, m, n, jumps, visited)*4;// 2,4,6,8 symmetric
		res+=numberOfPatternsUtil(5, 1, 0, m, n, jumps, visited);// start from 5
		return  res;
		
	}
	
	public int numberOfPatternsUtil(int num, int len, int count, int m, int n, int[][] jumps, boolean[] visited){
		if(len>=m)
			count++;
		len++;
		if(len>n)
			return count;
		visited[num]=true;
		
		for(int next=1;next<=9;next++){
			int jump = jumps[num][next];
			if(!visited[next]&&(jump==0||visited[jump]))
				count=numberOfPatternsUtil(next, len, count, m, n, jumps, visited);
		}
		visited[num]=false;
		return count;
	}
	
	
	public int numberOfPatterns2(int m, int n) {
		int[][] jumps=new int[10][10];
		boolean[] visited=new boolean[10];
		jumps[1][3]=jumps[3][1]=2;
		jumps[4][6]=jumps[6][4]=5;
		jumps[7][9]=jumps[9][7]=8;
		jumps[1][7]=jumps[7][1]=4;
		jumps[2][8]=jumps[8][2]=5;
		jumps[3][9]=jumps[9][3]=6;
		jumps[1][9]=jumps[9][1]=jumps[3][7]=jumps[7][3]=5;
		
		int res = 0;
		for(int i=m;i<=n;i++){
			res+=numberOfPatternsHelper(1, n-i, jumps, visited)*4;// 1,3,7, 9 symmetric
			res+=numberOfPatternsHelper(2, n-i, jumps, visited)*4;// 2,4,6,8 symmetric
			res+=numberOfPatternsHelper(5, n-i, jumps, visited);// start from 5
		}		
		return  res;
	}
	
	public int numberOfPatternsHelper(int cur, int remain, int[][] jumps, boolean[] visited){
		if(remain<0)
			return 0;
		if(remain==0)
			return 1;
		visited[cur]=true;
		int res = 0;
		for(int i=1;i<=9;i++){
			int jump=jumps[cur][i];
			if(!visited[i]&&(jump==0||visited[jump]))
				res+=numberOfPatternsHelper(i, remain-1, jumps, visited);
		}
		visited[cur]=false;
		return res;		
	}
	
	public int[] sortTransformedArray(int[] nums, int a, int b, int c) {
		int[] res = new int[nums.length];
		int i=0;
		int j=nums.length-1;
		int index=a>0?j:i;
		while(i<=j){
			int first = quad(a, b, c, nums[i]);
			int last = quad(a, b, c, nums[j]);
			if(a>0){
				if(first>last){
					res[index--]=first;
					i++;
				}else{
					res[index--]=last;
					j--;
				}
			}else{
				if(first>last){
					res[index++]=last;
					j--;
				}else{
					res[index++]=first;
					i++;
				}
			}
		}
		return res;
	}
	
	public int quad(int a, int b, int c, int x){
		return a*x*x+b*x+c;
	}
	
	public int[] sortTransformedArray2(int[] nums, int a, int b, int c) {
		int[] res = new int[nums.length];
		int i=0;
		int j=nums.length-1;
		int index=a>=0?j:i;
		
		while(i<=j){
			if(a>=0){
				res[index--]=quad(a,b,c,nums[i])>quad(a,b,c,nums[j])?quad(a,b,c,nums[i++]):quad(a,b,c,nums[j--]);
			}else{
				res[index++]=quad(a,b,c,nums[i])>quad(a,b,c,nums[j])?quad(a,b,c,nums[j--]):quad(a,b,c,nums[i++]);
			}
		}
		return res;
		
	}
	
	public boolean canConstruct(String ransomNote, String magazine) {
        int[] letters = new int[26];
        for(int i=0;i<magazine.length();i++){
            char c=magazine.charAt(i);
            letters[c-'a']++;
        }
        for(int i=0;i<ransomNote.length();i++){
            char c=ransomNote.charAt(i);
            if(--letters[c-'a']<0)
                return false;
        }
        return true;
    }
	
	public List<int[]> kSmallestPairs(int[] nums1, int[] nums2, int k) {
		List<int[]> res = new ArrayList<int[]>();
		if(nums1.length==0||nums2.length==0||k==0)
			return res;
		PriorityQueue<int[]> que=new PriorityQueue<int[]>(k, new Comparator<int[]>(){

			@Override
			public int compare(int[] o1, int[] o2) {
				// TODO Auto-generated method stub
				return o1[0]+o1[1]-o2[0]-o2[1];
			}
		});
		
		for(int i=0;i<nums1.length&&i<k;i++){
			que.add(new int[]{nums1[i], nums2[0], 0});
		}
		
		while(k-->0&&!que.isEmpty()){
			int[] cur=que.poll();
			res.add(new int[]{cur[0], cur[1]});
			if(cur[2]==nums2.length-1)
				continue;
			que.offer(new int[]{cur[0], nums2[cur[2]+1], cur[2]+1});
		}
		return res;
	}
	
	public ListNode plusOne(ListNode head) {
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode i=dummy, j=dummy;
		
		while(j!=null){
			if(j.val!=9){
				i=j;
			}
			j=j.next;
		}
		i.val++;
		i=i.next;
		while(i!=null){
			i.val=0;
			i=i.next;
		}
		if(dummy.val==0)
			return dummy.next;
		return dummy;
	}
	
	public ListNode plusOneRecur(ListNode head) {
		if(plusOneUtil(head)==0)
			return head;
		ListNode node=new ListNode(1);
		node.next=head;
		return node;
	}
	
	public int plusOneUtil(ListNode head){
		if(head==null)
			return 1;
		int sum = head.val+plusOneUtil(head.next);
		head.val=sum%10;
		return sum/10;
	}
	
	
	/*
	 * 从左上到右下的斜线上数字相同，并且matrix要是正方形，比如
1，5，3，4
0，1，5，3
9，0，1，5
2，9，0，1
Follow up: 假如matrix太大了，不能全部存在内存里，一次只能读一部分怎么办

	 */
	
	public boolean isSameDiagonal(int[][] matrix){
		int m=matrix.length;
		if(m==0)
			return true;
		int n = matrix[0].length;
		if(m!=n)
			return false;
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++){
				if((i+1)<m&&j+1<n&&matrix[i][j]!=matrix[i+1][j+1])
					return false;
			}
		}
		return true;
	}
	
	/*
	 * 有这样的一个List，list里面有好几个item：比如[(1,2,3), (1,2), (2,4), (2)]里面有4个item,每个item有一些数字（无duplicate）
把他partition，并且，分成的份数最少，规则是，他们必须要有相同的元素才能在一份中，比如这个例子可以分成：
[[(1,2,3), (1,2,),(2)], [(2,4)]] 或者[[(1,2,3), (1,2,)], [(2)，(2,4)]] 都是分成了两份。
	 */
	
	public  List<List<Set<Integer>>> getAnswer(List<Set<Integer>> input){
		List<List<Set<Integer>>> res = new ArrayList<List<Set<Integer>>>();
		Collections.sort(input, new Comparator<Set<Integer>>(){
			@Override
			public int compare(Set<Integer> o1, Set<Integer> o2) {
				return o2.size()-o1.size();
			}
		});
		List<Set<Integer>> lst=new ArrayList<Set<Integer>>();
		lst.add(input.get(0));
		res.add(lst);
		for(int i=1;i<input.size();i++){
			Set<Integer> set=input.get(i);
			boolean added=false;
			for(int j=0;j<res.size();j++){
				List<Set<Integer>> cur = res.get(j);
				for(int k=0;k<cur.size();k++){
					if(cur.get(k).containsAll(set)){
						cur.add(set);
						added=true;
						break;
					}
				}
				if(added)
					break;
			}
			if(!added){
				List<Set<Integer>> l=new ArrayList<Set<Integer>>();
				l.add(set);
				res.add(l);
			}
		}
		return res;
	}
	
	
	//给一个 二叉树 ， 求最深节点的最小公共父节点
	public TreeNode lcaOfDeepestNodes(TreeNode root){
		if(root==null)
			return null;
		int left=getTreeHeight(root.left);
		int right=getTreeHeight(root.right);
		if(left==right)
			return root;
		if(left>right)
			return lcaOfDeepestNodes(root.left);
		return lcaOfDeepestNodes(root.right);
		
	}
	
	
	public TreeNode lcaOfDeepestNodes2(TreeNode root){
		if(root==null)
			return null;
		Queue<TreeNode> que=new LinkedList<TreeNode>();
		que.add(root);
		Map<TreeNode, TreeNode> map = new HashMap<TreeNode, TreeNode>();
		List<TreeNode> level = new ArrayList<TreeNode>();
		List<TreeNode> lastlevel = new ArrayList<TreeNode>();
		int curlevel=1, nextlevel=0;
		while(!que.isEmpty()){
			TreeNode cur=que.poll();
			curlevel--;
			level.add(cur);
			if(cur.left!=null){
				que.add(cur.left);
				map.put(cur.left, cur);
				nextlevel++;
			}
			if(cur.right!=null){
				que.add(cur.right);
				map.put(cur.right, cur);
				nextlevel++;
			}
			if(curlevel==0){
				lastlevel=level;
				level = new ArrayList<TreeNode>();
				curlevel=nextlevel;
				nextlevel=0;
			}
		}
		if(lastlevel.size()==1)
			return lastlevel.get(0);
		TreeNode leftmost=lastlevel.get(0);
		TreeNode rightmost=lastlevel.get(lastlevel.size()-1);
		while(map.get(leftmost)!=map.get(rightmost)){
			leftmost=map.get(leftmost);
			rightmost=map.get(rightmost);
		}
		return map.get(rightmost);
	}
	
	/*
	 * 给你一个新的字母顺序规则，比如ole 表示o后面才可以出现l, l后面才可以出现 e. 给你一个string，满足这个字母顺序的的话返回true
	 * 否则返回false，不在这个规则顺序中的字母可以忽略. 就是说除了ole其他字母都不用管。
	 * 比如如下的string：
	Google 返回 true
	Elle 返回false（因为l有出现在e的后面）
	 */
	
	public boolean satisfyRules(String rule, String s){
		Map<Character, Integer> map=new HashMap<Character, Integer>();
		for(int i=0;i<rule.length();i++){
			map.put(rule.charAt(i), i);
		}
		char pre=' ';
		for(int i=0;i<s.length();i++){
			char cur=s.charAt(i);
			if(!map.containsKey(cur))
				continue;
			if(pre==' ')
				pre=cur;
			else{
				if(map.get(cur)<map.get(pre))
					return false;
				pre=cur;
			}
		}
		return true;
	}
	
	/*
	 * 给你一个 String和一个数字max,比如： “happy new year new year” max =2
	 * Max表示最长的可以有几个单词（要是在string中连续的单词）。返回所有的可能单词和次数 这个例子返回：
	 * 
	 * happy 1 
	 * new 2 
	 * year 2 
	 * happy new 1 
	 * new year 2 
	 * year new 1
	 */
	
	public Map<String, Integer> getAllStringsAndCounts(String s, int max){
		Map<String, Integer> res = new HashMap<String, Integer>();
		String[] ss=s.split(" ");
		for(int i=1;i<=max;i++){
			for(int start=0;start+i<=ss.length;start++){
				int end=start+i;
				String word = "";
				for(int j=start;j<end;j++){
					word+=ss[j]+" ";
				}
				word=word.trim();
				if(!res.containsKey(word))
					res.put(word, 1);
				else
					res.put(word, res.get(word)+1);
			}
		}
		return res;
	}
	
	/*
	 * 两个string，第二个比第一个多一个字母，找出这个字母。
比如 string1： abcd   string2 abecd   得出e
	 */
	
	public char diffChar(String s1, String s2){
		char[] chs1=s1.toCharArray();
		char[] chs2=s2.toCharArray();
		char res=0;
		for(char c: chs1){
			res^=c;
		}
		for(char c: chs2){
			res^=c;
		}
		return res;
	}
	
	public char findTheDifference(String s, String t) {
        char[] sc=s.toCharArray();
        char[] tc=t.toCharArray();
        char rs=0;
        for(char c: sc){
            rs^=c;
        }
        for(char c: tc){
            rs^=c;
        }
        return rs;
    }
	
	public char findTheDifference2(String s, String t) {
        int[] count=new int[26];
        for(char c: s.toCharArray()){
            count[c-'a']++;
        }
        for(char c: t.toCharArray()){
            if(--count[c-'a']<0)
            	return c;
        }
        return 0;
    }
	
	//two identical string except one string has one more char, how to find? --> what if shuffled?
	public char findExtraChar(String s1, String s2){
		if(s1.length()>s2.length()){
			String t=s1;
			s1=s2;
			s2=t;
		}
		for(int i=0;i<s1.length();i++){
			if(s1.charAt(i)!=s2.charAt(i))
				return s2.charAt(i);
		}
		return s2.charAt(s2.length()-1);
	}
	
	/*
	 * 一个数组中找出第一个Index，使得其左边的数字之和等于其右边所有数字之和；
Followup: 找出所有这样的indexes；找出一个数组中的最长递增子序列；找出一个二
叉树中从root到leaf所有path中的的最长递增子序列
	 */
	public int findIndex(int[] nums){
		if(nums.length<2)
			return -1;
		int sum = 0;
		for(int num: nums){
			sum+=num;
		}
		int cursum=0;
		for(int i=0;i<nums.length;i++){
			if(cursum==sum-nums[i]-cursum)
				return i;
			cursum+=nums[i];
		}
		return -1;
	}
	
	public int longestIncreasingSequence(int[] nums){
		if(nums.length<2)
			return nums.length;
		int[] dp=new int[nums.length];
		Arrays.fill(dp, 1);
		
		for(int i=1;i<nums.length;i++){
			for(int j=i-1;j>=0;j--){
				if(nums[j]<nums[i]&&dp[i]<dp[j]+1)
					dp[i]=dp[j]+1;
			}
		}
		int max=0;
		for(int i=0;i<dp.length;i++){
			max=Math.max(dp[i], max);
		}
		return max;
	}
	
	//给你一个多叉树。移出所有子树，当这个子树的全部节点的值的和为０
	public int removeSubtree(Node root){
		if(root==null)
			return 0;
		int sum=0;
		for(int i=0;i<root.children.size();i++){
			Node child = root.children.get(i);
			int subsum=removeSubtree(child);
			if(subsum==0)
				root.children.set(i, null);
			sum+=subsum;
		}
		sum+=root.val;
		return sum;
	}
	
	public int firstUniqChar(String s) {
        int[] count=new int[26];
        for(char c: s.toCharArray()){
            count[c-'a']++;
        }
        for(int i=0;i<s.length();i++){
            if(count[s.charAt(i)-'a']==1)
                return i;
        }
        return -1;
    }
	
	public NestedInteger deserializeNestedInteger(String s) {
		if(s.length()==0)
			return null;
		if(s.charAt(0)!='[')
			return new NestedIntegerImpl(Integer.valueOf(s));
		Stack<NestedIntegerImpl> stk = new Stack<NestedIntegerImpl>();
		int l=0;
		NestedIntegerImpl cur=null;
		for(int r=0;r<s.length();r++){
			if(s.charAt(r)=='['){
				if(cur!=null)
					stk.push(cur);
				cur = new NestedIntegerImpl();
				l=r+1;
			}else if(s.charAt(r)==']'){
				String num= s.substring(l, r);
				if(!num.isEmpty())
					cur.add(new NestedIntegerImpl(Integer.parseInt(num)));
				if(!stk.isEmpty()){
					NestedIntegerImpl ni = stk.pop();
					ni.add(cur);
					cur=ni;
				}
				l=r+1;
			}else if(s.charAt(r)==','){
				if(s.charAt(r-1)!=']'){
					String num = s.substring(l, r);
					cur.add(new NestedIntegerImpl(Integer.parseInt(num)));;
				}
				l=r+1;
			}
		}
		return cur;
	}
	
	public NestedInteger deserializeNestedInteger2(String s) {
		if(s.isEmpty())
			return null;
		if(s.charAt(0)!='[')
			return new NestedIntegerImpl(Integer.parseInt(s));
		Stack<NestedIntegerImpl> stk=new Stack<NestedIntegerImpl>();
		String cur="";
		for(int i=0;i<s.length();i++){
			char c=s.charAt(i);
			switch(c){
				case '[':
					stk.add(new NestedIntegerImpl());
					break;
				case ']':
					if(!cur.isEmpty()){
						stk.peek().add(new NestedIntegerImpl(Integer.parseInt(cur)));
						cur="";
					}
					if(stk.size()>1){
						NestedIntegerImpl ni = stk.pop();
						stk.peek().add(ni);
					}
					break;
				case ',':
					if(!cur.isEmpty()){
						stk.peek().add(new NestedIntegerImpl(Integer.parseInt(cur)));
						cur="";
					}
					break;
				default:
					cur+=c;
					break;
			}					
		}
		return stk.pop();
	}
	
	public NestedInteger deserializeRecursive(String s) {
		NestedInteger ni = new NestedIntegerImpl();
		if(s.isEmpty())
			return ni;
		if(s.charAt(0)!='[')
			return new NestedIntegerImpl(Integer.parseInt(s));
		else if (s.length()>2){
			int start=1, count=0;
			for(int i=1;i<s.length();i++){
				if(count==0&&(s.charAt(i)==','||i==s.length()-1)){
					ni.add(deserializeRecursive(s.substring(start, i)));
					start=i+1;
				}
				if(s.charAt(i)=='[')
					count++;
				if(s.charAt(i)==']')
					count--;
			}
		}
		return ni;
	}
	
	public int lengthLongestPath(String input) {
        int max = 0;
        String[] ss = input.split("\n");
        Stack<Integer> stk=new Stack<Integer>();
        int curlen=0;
        for(String s: ss){
        	int level = getLevel(s);
        	while(stk.size()>level){
        		curlen-=stk.pop();
        	}
        	int len=s.replace("\t", "").length()+1;
        	curlen+=len;
        	if(s.contains("."))
        		max=Math.max(max, curlen-1);
        	stk.push(len);
        }
        return max;
    }
	
	public int getLevel(String s){
		int count=0;
		for(int i=0;i<s.length();i++){
			if(s.charAt(i)=='\t')
				count++;
		}
		return count;
	}
	
	
	public int lengthLongestPath2(String input) {
		int max = 0;
		String[] ss = input.split("\n");
		Stack<String> stk = new Stack<String>();
		int curlen = 0;
		for (String s : ss) {
			if (stk.isEmpty() || getLevel(stk.peek()) < getLevel(s)) {
				stk.push(s);
				curlen += s.replace("\t", "").length() + 1;
			} else {
				while (!stk.isEmpty() && getLevel(stk.peek()) >= getLevel(s)) {
					curlen = curlen - stk.pop().replace("\t", "").length() - 1;
				}
				curlen += s.replace("\t", "").length() + 1;
				stk.push(s);
			}
			if (s.contains("."))
				max = Math.max(max, curlen - 1);
		}
		return max;
	}
	
	/*
	 * 给一个数字集合，数字是0-9，没有重复，输出由其中的数字构成的所有整数，该整数小于某一个特定的整数. 
	 * 比如：数字集合[1,2,3], 特定整数130。 输出: 1,2,3, 11,12,13,21,22,23,31,32,33, 111,112,113,121,122,123.
	 *  (下一个数字131 > 130), 输出的顺序无所谓。

	 */
	
	public List<Integer> findAllSmallerNumbers(int[] nums, int target){
		List<Integer> res=new ArrayList<Integer>();
		if(nums.length==0)
			return res;
		findAllSmallerNumbersUtil(nums, res, 0, target);
		return res;
	}
	
	public void findAllSmallerNumbersUtil(int[] nums, List<Integer> res, int curNum, int target){
		if(curNum>target)
			return;
		for(int i=0;i<nums.length;i++){
			curNum=curNum*10+nums[i];
			if(curNum<target)
				res.add(curNum);
			findAllSmallerNumbersUtil(nums, res, curNum, target);
			curNum=(curNum-nums[i])/10;
		}
	}
	
	//给一个xxx{xxx}xxx{xx}字符串，括号里面每次只能选一个字符，要求给出所有可能组合
	public List<String> findAllPossibleComs(String s){
		List<String> res = new ArrayList<String>();
		findAllPossibleComsUtil("", s, res);
		return res;
	}
	
	public void findAllPossibleComsUtil(String com, String s, List<String> res){
		int index1 = s.indexOf("{");
		if(index1<0){
			res.add(com+s);
		}else{
			int index2=s.indexOf("}");
			for(int i=index1+1;i<index2;i++){
				findAllPossibleComsUtil(com+s.substring(0, index1)+s.charAt(i), s.substring(index2+1), res);
			}
		}
	}
	
	//设计一个data structure来存储spare vector并做dot product
	public int DotProduct(Vector<Integer> A, Vector<Integer> B){
		List<int[]> la=new ArrayList<int[]>();
		List<int[]> lb=new ArrayList<int[]>();
		for(int i=0;i<A.size();i++){
			if(A.get(i)!=0){
				la.add(new int[]{i, A.get(i)});
			}
			if(B.get(i)!=0){
				lb.add(new int[]{i, B.get(i)});
			}
		}
		int i=0, j=0, res=0;
		while(i<la.size()&&j<lb.size()){
			if(la.get(i)[0]==lb.get(j)[0]){
				res+=la.get(i)[1]*lb.get(j)[1];
				i++;
				j++;
			}else if(la.get(i)[0]>lb.get(j)[0])
				j++;
			else
				i++;
		}
		return res;
	}
	
	//给一个array，按顺序输出发生重复的元素。举例：[1, 2, 3, 1, 1, 2, 4]，输出1和2。
	public List<Integer> findDuplicates(int[] A){
		List<Integer> res = new ArrayList<Integer>();
		Set<Integer> set1=new HashSet<Integer>();
		Set<Integer> set2=new HashSet<Integer>();
		for(int num :  A){
			if(!set1.contains(num)){
				set1.add(num);
			}else{
				if(!set2.contains(num)){
					set2.add(num);
					res.add(num);
				}
			}
		}
		return res;
	}
	
	/*
	 * 给一组国家到其人口的映射(map)：country->population，让求一个getCountry()
	 * 的函数，使返回的国家名字与其对应人口数成比例，可以使用内置的random函数。
	 * 
	 * 举例： 
	 * China -> 1.3 billion 
	 * US -> 0.4b billion 
	 * Russia -> 0.3b billion
	 * 
	 * 要求call getCountry 返回China的概率是65%，US的概率20%，Russia 的概率15%。
	 */
	
	class Country{
		String name;
		double start;
		double end;
		double population;
		public Country(String name, double start, double end, double population){
			this.name = name;
			this.start=start;
			this.end=end;
			this.population = population;
		}
		
		public String toString(){
			return name+"("+start+", "+end+")";
		}
	}
	
	public String getCountry(Map<String, Double> map){
		double sum = 0;
		Country[] countries = new Country[map.size()];
		int i = 0;
		for(String contry: map.keySet()){
			double pop = map.get(contry);
			countries[i++]=new Country(contry, sum, sum+pop, pop);
			sum+=pop;
		}
//		System.out.println(Arrays.toString(countries));
		double rand = sum * new Random().nextDouble();
//		System.out.println(rand);
		for(Country con: countries){
			if(rand>=con.start&&rand<con.end)
				return con.name;
		}
		return null;
	}
	
	public int largestBSTSubtree(TreeNode root) {
		int[] max={0};
		largestBSTSubtreeUtil(root, max);
		return max[0];
	}
	
	public SuperNode largestBSTSubtreeUtil(TreeNode root, int[] max){
		if(root==null)
			return new SuperNode(0, Integer.MAX_VALUE, Integer.MIN_VALUE);
		SuperNode left=largestBSTSubtreeUtil(root.left, max);
		SuperNode right=largestBSTSubtreeUtil(root.right, max);
		if(left.size==-1||right.size==-1||left.upper>=root.val||right.lower<=root.val)
			return new SuperNode(-1, 0, 0);
		int size = left.size+1+right.size;
		max[0]=Math.max(size, max[0]);
		return new SuperNode(size, Math.min(root.val, left.lower), Math.max(root.val, right.upper));
	}
	
	class Data{
		boolean isBST;
		int size;
		int lower;
		int upper;
	}
	
	public int largestBSTSubtree2(TreeNode root) {
		int[] max={0};
		largestBSTSubtreeUtil2(root, max);
		return max[0];
	}
	
	public Data largestBSTSubtreeUtil2(TreeNode root, int[] max){
		Data cur=new Data();
		if(root==null){
			cur.isBST=true;
			cur.size=0;
			return cur;
		}
		Data left=largestBSTSubtreeUtil2(root.left, max);
		Data right=largestBSTSubtreeUtil2(root.right, max);
		cur.lower=Math.min(root.val, Math.min(left.lower, right.lower));
		cur.upper=Math.max(root.val, Math.max(left.upper, right.upper));
		
		if(left.isBST&&right.isBST&&root.val>left.upper&&root.val<right.lower){
			cur.isBST=true;
			cur.size=left.size+1+right.size;
			max[0]=Math.max(max[0], cur.size);
		}else{
			cur.size=0;
			cur.isBST=false;
		}
		return cur;
	}
	
//	Largest subarray with equal number of 0s and 1s
	public int maxLen(int[] A){
		Map<Integer, Integer> map=new HashMap<Integer, Integer>();
		for(int i=0;i<A.length;i++){
			if(A[i]==0)
				A[i]=-1;
		}
		int maxlen=0;
		int endIndex=-1;
		int sum = 0;
		for(int i=0;i<A.length;i++){
			sum+=A[i];
			if(sum==0){
				maxlen=i+1;
				endIndex=i;
			}
			if(map.containsKey(sum)){
				if(i-map.get(sum)>maxlen){
					maxlen=i-map.get(sum);
					endIndex=i;
				}
			}else
				map.put(sum, i);
		}
		for(int i=0;i<A.length;i++){
			if(A[i]==-1)
				A[i]=0;
		}
		int start = endIndex-maxlen+1;
		System.out.println("from "+start+" to "+endIndex);
		return maxlen;
	}
	
	public boolean isWordSquare(char[][] words){
		for(int i=0;i<words.length;i++){
			for(int j=i;j<words[0].length;j++){
//				System.out.println(words[i][j]+"<---->"+words[j][i]);
				if(words[i][j]!=words[j][i])
					return false;
			}
		}
		return true;
	}
	
//	public List<List<String>> findAllWordSquare(List<String> words, int len){
//	List<List<String>> res = new ArrayList<List<String>>();
//	for(int i=0;i<words.size();i++){
//		String word=words.get(i);
//		for(int j=0;j<word.length();j++){
//			
//		}
//	}
//}
	
	public boolean isSubsequence(String s, String t) {
        if(s.length()>t.length())
        	return false;
        int i=0, j=0;
        while(i<s.length()&&j<t.length()){
        	if(s.charAt(i)==t.charAt(j))
        		i++;
        	j++;
        }
        return i==s.length();
    }
	
	//product of the array   给定一个array，返回里面元素乘积的所有可能值。 例如给定array:[1,2,3,4] 
	//应该返回 [1, 2, 3, 4, 6, 8, 12, 24]
	public List<Integer> findAllPossibleProducts(int[] A){
		List<Integer> res = new ArrayList<Integer>();
		findProductsUtil(A, 0, res, 1);
		return res;
	}
	
	public void findProductsUtil(int[] A, int cur, List<Integer> res, int prod){
		if(cur==A.length)
			return;
		for(int i=cur;i<A.length;i++){
			prod*=A[i];
			if(!res.contains(prod))
				res.add(prod);
			findProductsUtil(A, i+1, res, prod);
			prod/=A[i];
		}
	}
	
	// need to remove 1 if 1 does not exit in array
	public List<Integer> product(int[] nums) {
        List<Integer> res = new ArrayList<Integer>();
        Set<Integer> set = new HashSet<Integer>();
        dfs(nums, 0, 1, res, set);
        Collections.sort(res);
        return res;
    }
    
    private void dfs(int[] nums, int start, int product, List<Integer> res, Set<Integer> set) {
        if (set.contains(product)) return;
        
        set.add(product);
        res.add(product);
        
        for (int i = start; i < nums.length; i++) {
            product *= nums[i];
            dfs(nums, i + 1, product, res, set);
            product /= nums[i];
        }
    }
	
	//subarray sum to K      在一个array中，找出是否有连续的subarray sums to K，返回true or false
	public boolean subarraySumToK(int[] A, int k){
		Set<Integer> set=new HashSet<Integer>();
		int sum = 0;
		for(int num: A){
			sum+=num;
			if(sum==k)
				return true;
			if(set.contains(sum-k))
				return true;
			set.add(sum);
		}
		return false;
	}
	
//	一个Sorted array,给一个Target, 找到最大的index 使得Array[index] == target.
	public int findIndex(int[] A, int target){
		int i=0, j=A.length-1;
		int index=-1;
		while(i<=j){
			int mid=(i+j)/2;
			if(A[mid]==target){
				index=mid;
				break;
			}
			if(A[mid]>target)
				j=mid-1;
			else
				i=mid+1;
		}
		if(index==-1)
			return -1;
		i=index;
		j=A.length-1;
		while(i<=j){
			int mid=(i+j)/2;
			if(A[mid]>target)
				j=mid-1;
			else
				i=mid+1;
		}
		return j;
	}
	
	public int findIndex2(int[] A, int target){
		int i=0, j=A.length-1;
		while(i<j){
			int mid=(i+j)/2+1;
			if(A[mid]>target)
				j=mid-1;
			else
				i=mid;
		}
		if(A[j]==target)
			return j;
		return -1;
	}
	
	public String decodeString(String s) {
        if(s.isEmpty()||!s.contains("[")||!s.contains("]"))
        	return s;
        int index=s.indexOf('[');
        int idx=index-1;
        while(idx>=0&&s.charAt(idx)>='0'&&s.charAt(idx)<='9'){
        	idx--;
        }
        int times = Integer.parseInt(s.substring(idx+1, index));
        int count=1;
        int i=index+1;
        for(;i<s.length();i++){
        	if(s.charAt(i)=='[')
        		count++;
        	else if(s.charAt(i)==']')
        		count--;
        	if(count==0)
        		break;
        }
        StringBuilder sb=new StringBuilder();
        sb.append(s.substring(0, idx+1));
        for(int k=0;k<times;k++){
        	sb.append(s.substring(index+1, i));
        }
        sb.append(s.substring(i+1));
        return decodeString(sb.toString());
    }
	
	
	public String serializeTree(TreeNode root){
		if(root==null)
			return "";
		StringBuilder sb =new StringBuilder();
		serializeTreeHelper(root, sb);
		return sb.toString();
	}
	
	public void serializeTreeHelper(TreeNode root, StringBuilder sb){
		if(root==null)
			return;
		sb.append("(");
		sb.append(root.val);
		serializeTreeHelper(root.left, sb);
		serializeTreeHelper(root.right, sb);
		sb.append(")");
	}
	
//	public TreeNode deserializeTree(String)
	

	public static void main(String[] args) throws UnknownHostException,
			IOException {
		Solutions sol = new Solutions();

		 TreeNode root = new TreeNode(1);
		 root.left = new TreeNode(2);
		 root.right = new TreeNode(0);
		 root.left.left = new TreeNode(3);
		 root.left.right = new TreeNode(4);
		 root.right.left = new TreeNode(5);
//		 root.right.right = new TreeNode(6);
//		 root.right.right.left = new TreeNode(12);

		 System.out.println(sol.serializeTree(root));
//		TreeNode root = new TreeNode(8);
//		root.left = new TreeNode(1);
//		root.right = new TreeNode(2);
//		// root.left.left = new TreeNode(2);
//		root.left.right = new TreeNode(0);
//		root.right.left = new TreeNode(3);
//		root.right.right = new TreeNode(4);
//		root.right.right.left = new TreeNode(5);
		
		System.out.println(sol.lcaOfDeepestNodes(root).val);
		System.out.println(sol.lcaOfDeepestNodes2(root).val);
		System.out.println(sol.serialize2(root));

		System.out.println(sol.findLeaves(root));
		System.out.println(sol.levelOrderInterval(root));

		System.out.println(sol.inorderTraversal(root));
		sol.sinkZeros(root);
		System.out.println(sol.inorderTraversal(root));

		TreeNode halfRoot = sol.removeHalfNodes(root);
		sol.inorder(halfRoot);

		System.out.println(sol.closestKValues(root, 5, 4));

		System.out.println();
		TreeNode p = sol.UpsideDownBinaryTree(root);
		sol.inorder(p);
		System.out.println();

		List<Pair> list = new ArrayList<Pair>();
		Pair p1 = new Pair('1', '2');
		Pair p2 = new Pair('1', '3');
		Pair p3 = new Pair('1', '4');
		Pair p4 = new Pair('2', '5');
		Pair p5 = new Pair('2', '6');
		Pair p6 = new Pair('3', '8');
		Pair p7 = new Pair('6', '9');
		list.add(p1);
		list.add(p2);
		list.add(p3);
		list.add(p4);
		list.add(p5);
		list.add(p6);
		list.add(p7);

		Node node = sol.buildTree(list);

		System.out.println(sol.printTree(node));

		Point r1 = new Point(0, 3);
		Point r2 = new Point(1, 2);
		Point r3 = new Point(2, 4);
		Point r4 = new Point(1, 3);
		Point r5 = new Point(1, 0);
		Point r6 = new Point(3, 2);
		List<Point> l = new ArrayList<Point>();
		l.add(r6);
		l.add(r5);
		l.add(r4);
		l.add(r3);
		l.add(r2);
		l.add(r1);
		System.out.println(sol.raceRanking(l));

		int[] number = { 3, 5, 8, 2, 9, 7, 9, 4 };
		int[] countGreaterThan = { 1, 0, 0, 7, 0, 3, 0, 5 };
		sol.arrangeHeight(number, countGreaterThan);

		System.out.println(sol.generatePalindromes("aabb"));

		int[][] edegs = { { 1, 0 }, { 1, 2 }, { 3, 4 } };
		// System.out.println(sol.findMinHeightTrees(5, edegs));
		System.out.println(sol.countComponents(5, edegs));
		System.out.println(sol.countComponents2(5, edegs));
		int[] nums1 = { 4, 9, 5 };
		int[] nums2 = { 8, 7, 4 };
		sol.merge(nums1, nums2, 5);
		sol.maxNumber(nums1, nums2, 3);

		int[][] grid = { { 1, 0, 1, 1 }, { 0, 0, 0, 1 }, { 1, 1, 0, 1 } };
		System.out.println(sol.minTotalDistance4(grid));

		List<Integer> nums = new ArrayList<Integer>(Arrays.asList(1, 2, 2, 3,
				3, 4, 4, 5, 5));
		System.out.println(sol.groupNumbers(nums));

		List<Integer> coins = new ArrayList<Integer>(Arrays.asList(1, 2, 3, 4));
		System.out.println(sol.coinsSums(coins));

		System.out.println(sol.longestPalindromeGoogle("abccbdad"));

		System.out.println(sol.getAllLongestPanlindromes("aaabbbc"));

		char[][] board = { { 'x', 'x', 'x', 'x' }, { 'x', 'o', 'o', 'x' },
				{ 'x', 'x', 'o', 'x' }, { 'x', 'x', 'x', 'x' } };
		System.out.println(sol.canSurvive(board, 1, 2));

		Event e1 = new Event(5, 9);
		Event e2 = new Event(12, 15);
		Event e3 = new Event(26, 30);
		Event e4 = new Event(11, 13);

		Event[] events = { e1, e2, e3, e4 };

		System.out.println(sol.longestOccupyTime(events));

		Map<String, String> morseCode = new HashMap<String, String>();
		morseCode.put("A", ".-");
		morseCode.put("B", "-...");
		morseCode.put("C", "-.-.");
		morseCode.put("D", "-..");
		morseCode.put("E", ".");
		morseCode.put("F", "..-.");
		morseCode.put("G", "--.");
		morseCode.put("H", "....");
		morseCode.put("I", "..");

		Map<String, String> inverseMorseCode = new HashMap<String, String>();
		for (Map.Entry<String, String> pair : morseCode.entrySet()) {
			inverseMorseCode.put(pair.getValue(), pair.getKey());
		}

		String morse = sol.encodeMorse("ABC", morseCode);
		System.out.println(morse);

		System.out.println(sol.decodeMorse(morse.trim(), inverseMorseCode));

		ArrayList<Node> nodes = new ArrayList<Node>();
		nodes.add(new Node(10, 30, 1));
		nodes.add(new Node(30, 0, 10));
		nodes.add(new Node(20, 30, 2));
		nodes.add(new Node(50, 40, 3));
		nodes.add(new Node(40, 30, 4));
		sol.printSubTreeWeight(nodes);
		System.out.println();
		sol.printSubTreeWeight2(nodes);

		int arr[] = { 1, 2, 1, 3, 4, 2, 3 };
		sol.countDistinct(arr, 4);

		String[] friends = { "ynyy", "nyyn", "yyyn", "ynny" };
		System.out.println(sol.friendCirlce(friends));
		System.out.println(sol.friendCirclesBFS(friends));
		String[] words2 = { "a", "abcd", "bcd", "abd", "cd", "c" };
		System.out.println(sol.longestChain(words2));

		String path = "dir\n ddir\n  a.txt\n  b.jpeg\n  cdef.gif\n  dddir\n   a.png\n ddir2\n dddir\n ddddir\n  aa.exe\n";

		System.out.println(sol.longestImagePath(path));
		System.out.println(sol.getFactors(32));
		System.out.println(sol.getAllFactors(35));

		System.out.println(sol.findMissingNumberGivenString("88899100"));

		int[] sort = { 5, 2, 7, 8, 1, 4, 3 };
		sol.quicksort(sort);

		int[][] matrix = { { 1, 2, -1, -4, -20 }, { -8, -3, 4, 2, 1 },
				{ 3, 8, 10, 1, 3 }, { -4, -1, 1, 7, -6 } };

		System.out.println(sol.findMaxSubMatrixSum(matrix));

		int[][] edges = { { 0, 1 }, { 1, 2 }, { 2, 3 }, { 1, 3 }, { 1, 4 } };
		System.out.println(sol.validTreeDFS(5, edges));

		Ads ad1 = new Ads(2, 3, 5);
		Ads ad2 = new Ads(1, 4, 10);
		Ads ad3 = new Ads(4, 5, 7);
		Ads ad4 = new Ads(4, 7, 4);
		Ads ad5 = new Ads(5, 7, 8);

		Ads[] ads = { ad1, ad2, ad3, ad4, ad5 };
		System.out.println(sol.maxAdsProfit(ads, 10));
		System.out.println(sol.maxAdsProfit2(ads, 10));
		System.out.println(sol.maxAdsProfit(ads));
		System.out.println(sol.maxAdsProfitFollowup(ads));

		int[] X = { 2, 3, 5 };
		System.out.println(sol.getSumExcludePrimes(10, X));

		System.out.println(sol.getAllPermutations("AAABBC", 3));

		int[] prod = { 2, -3, 3, -2, -4, 3 };
		System.out.println(sol.KMaxProductOfTwo(prod, 3));

		System.out.println(sol.wordAbbreviations("world"));
		System.out.println(sol.generateAbbreviations("world"));

		String[] strs = { "gab", "cat", "bag", "alpha", "cc", "gab" };
		System.out.println(sol.getPalindromaticPairs(strs));

		// Interval i1 = new Interval(1, 3);
		// Interval i2 = new Interval(2, 3);
		// Interval i3 = new Interval(1, 4);
		// Interval i4 = new Interval(5, 6);
		// Interval i5 = new Interval(2, 4);
		// Interval i6 = new Interval(7, 8);
		// Interval[] intervals = { i1, i2, i3, i4, i5, i6 };
		// System.out.println(sol.findFreeInterval(intervals));
		int[] days = { 5, 1, 1, 5 };
		System.out.println(sol.getMaxRentalDays(days));

		double[] A = { 1.6, 1.6, 1.6 };
		System.out.println(Arrays.toString(sol.rounding(A)));

		String[] words3 = { "wrt", "wrf", "er", "ett", "rftt" };
		System.out.println(sol.alienOrderDFS(words3));
		System.out.println(sol.topologicalSort(words3));

		Point pro1 = new Point(1, 1);
		Point pro2 = new Point(2, 2);
		Point end1 = new Point(1, 3);
		Point pro3 = new Point(3, 3);
		Point end2 = new Point(2, 4);
		Point pro4 = new Point(4, 5);
		Point end4 = new Point(4, 6);
		Point end3 = new Point(3, 7);

		Point[] starts = { pro1, pro2, pro3, pro4 };
		Point[] ends = { end1, end2, end4, end3 };

		sol.printProcessOrder(starts, ends);

		System.out.println(sol.matching("abcde", "*d*e*"));
		System.out.println(sol.matching2("abcde", "*d*e*"));

		int[] A1 = { 1, 2, 2, 3, 3, 3, 4 };
		System.out.println(sol.getLastIndex(A1, 2));

		int[] money = { 3, 2, 2, 3, 1, 2 };
		System.out.println(sol.maxMoneyFollowup(money));

		String[] words = { "acde", "abc", "abd", "ad" };
		System.out.println(sol.getKEditDistance(words, "ace", 1));

		Set<Character> chars = new HashSet<Character>();
		chars.add('a');
		chars.add('p');
		System.out.println(sol.generataAllStringFollowUp("airpplane", chars));

		int[] numbers = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 };
		System.out.println(Arrays.toString(sol.reverseMatrix(4, 3, numbers)));

		String[] words4 = { "a", "" };
		System.out.println(sol.palindromePairs(words4));

		Set<String> dict = new HashSet<String>();
		dict.add("i");
		dict.add("a");
		dict.add("am");
		dict.add("happy");
		dict.add("hello");
		dict.add("after");
		dict.add("noon");
		dict.add("afternoon");

		System.out.println(sol.wordInDicionary("helloafternoon", dict));

		int[][] dots = { { 0, 1, 2 }, { 3, 4, 5 }, { 6, 7, 8 } };

		System.out.println(sol.validPin(dots, "14752086"));

		System.out.println(sol.countWaysOfTiling(3));
		System.out.println(sol.countWaysOfTiling(4));
		System.out.println(sol.countWaysOfTilingFollowUp(5));

		System.out.println(sol.fractionToDecimal2(13, 6));

		System.out.println(sol.splitLotteryNumbers("122345678"));
		int[] weight = { 2, 5, 6 };
		System.out.println(sol.nextFit(weight, 10));
		System.out.println(sol.firstFit(weight, 10));

		System.out.println(sol.printAllPossiblePalindromes("aabbcadad"));

		int[] R = { 3, 2, 3, 4, 4, 5, 2, 3, 2, 4, 5 };
		System.out.println(sol.highestSkillRating(11, R, 4));

		List<Boolean> flowerbed = new ArrayList<Boolean>();
		// flowerbed.add(true);
		flowerbed.add(false);
		flowerbed.add(true);
		System.out.println(sol.canPlaceFlowers(flowerbed, 1));

		System.out.println(sol.findNumbersWithNBits(3, 2));

		int[] c = { 2, 5, 7, 3 };
		System.out.println(sol.coinChangeFollowUp(c, 13));

		boolean[][] grids = { { false, false, true, false },
				{ false, true, false, true }, { false, false, false, false },
				{ false, false, true, false } };
		System.out.println(sol.countSquares(grids));

		Node naryRoot = new Node('1');
		List<Node> children1 = new ArrayList<Node>();
		Node ch1 = new Node('3');
		Node ch2 = new Node('2');
		Node ch3 = new Node('4');
		Node ch4 = new Node('5');
		children1.add(ch1);
		children1.add(ch2);
		children1.add(ch3);
		children1.add(ch4);
		naryRoot.children = children1;
		List<Node> children2 = new ArrayList<Node>();
		Node ch21 = new Node('7');
		children2.add(ch21);
		ch2.children = children2;
		List<Node> children3 = new ArrayList<Node>();
		Node ch22 = new Node('5');
		children3.add(ch22);
		ch3.children = children3;
		List<Node> children4 = new ArrayList<Node>();
		Node ch23 = new Node('6');
		children4.add(ch23);
		ch4.children = children4;

		List<Node> children5 = new ArrayList<Node>();
		Node ch31 = new Node('6');
		children5.add(ch31);
		ch22.children = children5;

		// Node ch32=new Node('f');
		// Node ch33=new Node('h');
		// Node ch34=new Node('a');
		// Node ch41=new Node('a');
		// Node ch42=new Node('a');

		System.out.println(sol.longestConsecutiveFollowup(naryRoot));

		ListNodeWithDown downHead = new ListNodeWithDown(1);
		downHead.next = new ListNodeWithDown(2);
		downHead.next.next = new ListNodeWithDown(3);
		downHead.next.next.next = new ListNodeWithDown(4);
		downHead.next.next.next.next = new ListNodeWithDown(5);
		downHead.next.down = new ListNodeWithDown(6);
		downHead.next.down.next = new ListNodeWithDown(7);
		downHead.next.down.next.next = new ListNodeWithDown(8);
		downHead.next.down.down = new ListNodeWithDown(9);
		downHead.next.down.down.next = new ListNodeWithDown(10);

		ListNodeWithDown down = sol.flattenListRecursive(downHead);
		while (down != null) {
			System.out.print(down.val + " ");
			down = down.next;
		}

		ListNode head1 = new ListNode(9);
		head1.next = new ListNode(2);
		head1.next.next = new ListNode(7);

		ListNode head2 = new ListNode(1);
		head2.next = new ListNode(3);
		// head2.next.next=new ListNode(3);

		ListNode newHead = sol.LinkedListAddtionForwardOrder(head1, head2);
		while (newHead != null) {
			System.out.print(newHead.val + " ");
			newHead = newHead.next;
		}
		System.out.println();

		int[][] mat = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 },
				{ 13, 14, 15, 16 }, { 17, 18, 19, 20 } };
		System.out.println(sol.printDiagonalElements(mat));
		System.out.println(sol.printDiagonalElements2(mat));

		int[] seq = { 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 200, 201, 202,
				203, 204, 205 };
		System.out.println(sol.longestSubsequence(seq));

		int[][] graph = { { 0, 16, 13, 0, 0, 0 }, { 0, 0, 10, 12, 0, 0 },
				{ 0, 4, 0, 0, 14, 0 }, { 0, 0, 9, 0, 0, 20 },
				{ 0, 0, 0, 7, 0, 4 }, { 0, 0, 0, 0, 0, 0 } };
		sol.minCut(graph, 0, 5);

		int[] numbs = { 5, 8, 1, 3, 4, 2, 7 };
		System.out.println(sol.countSmallerBefore(numbs));
		sol.wiggleSort2(numbs);

		int graph2[][] = new int[][] { { 0, 2, 0, 6, 0 }, { 2, 0, 3, 8, 5 },
				{ 0, 3, 0, 0, 7 }, { 6, 8, 0, 0, 9 }, { 0, 5, 7, 9, 0 }, };
		sol.primMST(graph2);

		sol.splitWords("My name is \" \" and \"Donald Trump\"");
		sol.splitWords(" \" ");

		int[][] m = { { 4, 8, 7, 3 }, { 2, 5, 9, 3 }, { 6, 3, 2, 5 },
				{ 4, 4, 1, 6 } };
		System.out.println(sol.getLongestPath(m));

		int[] seq1 = { 4, 2, 8, 10, 11 };
		System.out.println(sol.longestConsecutiveSequence(seq1));

		Interval in1 = new Interval(0, 10);
		Interval in2 = new Interval(10, 15);
		Interval in3 = new Interval(13, 20);
		Interval in4 = new Interval(0, 5);
		Interval in5 = new Interval(27, 33);
		Interval in6 = new Interval(21, 24);
		List<Interval> sch1 = Arrays.asList(in1, in2, in3);
		List<Interval> sch2 = Arrays.asList(in4, in6, in5);

		System.out.println(sol.findLatestStartTime(sch1, sch2, 2));
		System.out.println(sol.mergeIntervals(sch1, sch2));
		System.out.println(sol.findAllPossibleCombinations3("01?1"));
		System.out.println(sol
				.lengthOfLongestSubstringKDistinct("ABACBAAAB", 2));

		System.out.println(sol.rearrange("aaaabbbcc", 2));
		System.out.println(sol.subtract("101", "231"));

		System.out.println(sol.letterCombinations("321"));

		int[] topk = { 1, 1, 1, 2, 2, 3 };
		System.out.println(sol.topKFrequent(topk, 2));

		int[] nn = { -3, 3 };
		System.out.println(sol.permuteIterative2(nn));
		System.out.println("-----------------------------");
		System.out.println(sol.containsNearbyAlmostDuplicate2(nn, 2, 4));

		Interval i1 = new Interval(1, 5);
		Interval newInterval = new Interval(2, 3);
		List<Interval> intervals = new ArrayList<Interval>(Arrays.asList(i1));

		System.out.println(sol.insert2(intervals, newInterval));

		System.out.println(sol.getAllLongestPanlindromes("ssssbba"));

		System.out.println(sol.largeNumDivide("1000", "100", 5));
		System.out.println(sol.longestValidParentheses2(")("));

		int[][] grid2 = { { 1, 0, 2, 0, 1 }, { 0, 0, 0, 0, 0 },
				{ 0, 0, 1, 0, 0 } };
		System.out.println(sol.shortestDistance(grid2));
		int[] citations = { 6, 5, 4, 3, 1, 0 };
		System.out.println(sol.hIndexII(citations));

		String[][] tickets = { { "NY", "DC" }, { "DC", "LA" }, { "NY", "LA" },
				{ "Chicago", "DC" }, { "LA", "NY" }, { "NY", "Chicago" } };

		System.out.println(sol.findAllRoutes("DC", "NY", tickets));
		// [[2,3],[3,3],[-5,3]]
		Point[] points = { new Point(2, 3), new Point(3, 3), new Point(-5, 3) };

		System.out.println(sol.maxPoints3(points));

		String s1 = "abc";
		String s2 = "abc";
		System.out.println(s1.equals(s2));

		char[][] gridEnemy = { { '0', 'E', '0', '0' }, { 'E', '0', 'W', 'E' },
				{ '0', 'E', '0', '0' } };

		System.out.println(sol.maxKilledEnemies(gridEnemy));
		System.out.println(sol.maxKilledEnemies2(gridEnemy));
		
		int[] nums100={1,2,3};
		System.out.println(sol.combinationSum4(nums100, 4));
		
		System.out.println(sol.numberOfPatterns(1, 1));
		System.out.println(sol.numberOfPatterns2(1, 1));
		System.out.println(sol.numberOfPatterns(1, 2));
		System.out.println(sol.numberOfPatterns2(1, 2));

		int[] nums200={-5,-3,-1,0,1,3,5,7};
		System.out.println(Arrays.toString(sol.sortTransformedArray(nums200, -1, -1, 0)));
		System.out.println(Arrays.toString(sol.sortTransformedArray2(nums200, -1, -1, 0)));
		
		int[][] mm={{1,5,3,4},
				{0,1,5,3},
				{9,0,1,5},
				{2,9,0,1}};
		
		System.out.println(sol.isSameDiagonal(mm));
		
		List<Set<Integer>> inputs = new ArrayList<Set<Integer>>();
		Set<Integer> set1 = new HashSet<Integer>();
		Set<Integer> set2 = new HashSet<Integer>();
		Set<Integer> set3 = new HashSet<Integer>();
		Set<Integer> set4 = new HashSet<Integer>();
		set1.add(1);
		set1.add(2);
		set1.add(3);
		set2.add(1);
		set2.add(2);
		set3.add(2);
		set3.add(4);
		set4.add(2);
		
		inputs.add(set1);
		inputs.add(set2);
		inputs.add(set3);
		inputs.add(set4);
		
		System.out.println(sol.getAnswer(inputs));
		
		ListNode head11 = new ListNode(1);
		head11.next=new ListNode(9);
		head11.next.next=new ListNode(8);
		
		ListNode nhead=sol.plusOne(head11);
		while(nhead!=null){
			System.out.print(nhead.val+" ");
			nhead=nhead.next;
		}
		System.out.println();
		nhead=sol.plusOneRecur(head11);
		while(nhead!=null){
			System.out.print(nhead.val+" ");
			nhead=nhead.next;
		}
		System.out.println();
		System.out.println(sol.satisfyRules("ole", "google"));
		System.out.println(sol.satisfyRules("ole", "elle"));
		System.out.println(sol.getAllStringsAndCounts("happy new year new year", 3));
		System.out.println(sol.diffChar("abcd", "abcde"));
		
		int[] nnums={10, 22, 9, 33, 21, 50, 41, 60, 18};
		System.out.println(sol.findIndex(nnums));
		System.out.println(sol.longestIncreasingSequence(nnums));
		
		System.out.println(sol.lengthLongestPath("dir\n\tsubdir1\n\tsubdir2\n\t\tfile.ext"));
		System.out.println(sol.lengthLongestPath2("dir\n\tsubdir1\n\t\tfile1.ext\n\t\tsubsubdir1\n\tsubdir2\n\t\tsubsubdir2\n\t\t\tfile2.ext"));
	
		int[] nnnums={1,2,3};
		System.out.println(sol.findAllSmallerNumbers(nnnums, 130));
		
		char[][] ws={{'a','b','c','d'},{'b', 'n','r', 't'},{'c','r','m', 'y'},{'d','t','y', 'x'}};
		
		System.out.println(sol.isWordSquare(ws));
		System.out.println(sol.findAllPossibleComs("abc{de}"));
		
		int[] tttt={1, 2, 3, 1, 1, 2, 4};
		System.out.println(sol.findDuplicates(tttt));
		
		Map<String, Double> countries = new HashMap<String, Double>();
		countries.put("China", 1.3);
		countries.put("USA", 0.3);
		countries.put("Russia", 0.4);
		
		System.out.println(sol.getCountry(countries));
		System.out.println(sol.findExtraChar("123", "0123"));
		
		int[] zeroOnes = {0, 0, 1, 1, 0};
		System.out.println(sol.maxLen(zeroOnes));
		
		int[] pt={2,4, 7, -1, -4, 2};
		System.out.println(sol.findAllPossibleProducts(pt));
		System.out.println(sol.product(pt));
		System.out.println(sol.subarraySumToK(pt, 10));
		
		int[] sortedA={1,2,3,3,3,3,4,4,4,5,5,5,6};
		
		System.out.println(sol.findIndex(sortedA, 3));
		System.out.println(sol.findIndex2(sortedA, 3));
		
		System.out.println(sol.decodeString("3[a2[c]]"));
	}
}
