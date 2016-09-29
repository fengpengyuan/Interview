package com.lintcode;

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
import java.util.Set;
import java.util.Stack;

public class Solutions {

	class TrieNode {
		boolean isString;
		HashMap<Character, TrieNode> substree;
		String s;

		public TrieNode() {
			isString = false;
			substree = new HashMap<Character, TrieNode>();
			s = "";
		}

	}

	class TrieTree {
		TrieNode root;

		public TrieTree(TrieNode TrieNode) {
			root = TrieNode;
		}

		public void insert(String s) {
			TrieNode now = root;
			for (int i = 0; i < s.length(); i++) {
				if (!now.substree.containsKey(s.charAt(i))) {
					now.substree.put(s.charAt(i), new TrieNode());
				}
				now = now.substree.get(s.charAt(i));
			}
			now.s = s;
			now.isString = true;
		}

		public boolean find(String s) {
			TrieNode now = root;
			for (int i = 0; i < s.length(); i++) {
				if (!now.substree.containsKey(s.charAt(i)))
					return false;
				now = now.substree.get(s.charAt(i));
			}
			return now.isString;
		}
	}

	public ArrayList<String> wordSearchII2(char[][] board,
			ArrayList<String> words) {
		ArrayList<String> res = new ArrayList<String>();
		TrieNode root = new TrieNode();
		TrieTree tree = new TrieTree(root);

		for (String word : words) {
			tree.insert(word);
		}
		boolean[][] used = new boolean[board.length][board[0].length];
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board[0].length; j++) {
				findAllWords(board, i, j, used, tree.root, res);
			}
		}
		return res;
	}

	public void findAllWords(char[][] board, int i, int j, boolean[][] used,
			TrieNode root, ArrayList<String> res) {
		if (root.isString) {
			if (!res.contains(root.s))
				res.add(root.s);
		}
		if (i < 0 || i >= board.length || j < 0 || j >= board[0].length
				|| used[i][j] || root == null)
			return;
		int[] dx = { 1, -1, 0, 0 };
		int[] dy = { 0, 0, 1, -1 };
		if (root.substree.containsKey(board[i][j])) {
			used[i][j] = true;
			for (int k = 0; k < 4; k++) {
				findAllWords(board, i + dx[k], j + dy[k], used,
						root.substree.get(board[i][j]), res);
			}
			used[i][j] = false;
		}
	}

	public int[] twoSum(int[] numbers, int target) {
		// write your code here
		int[] res = { -1, -1 };
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < numbers.length; i++) {
			map.put(numbers[i], i + 1);
		}

		for (int i = 0; i < numbers.length; i++) {
			int num = numbers[i];
			if (map.containsKey(target - num)) {
				res[0] = i + 1;
				res[1] = map.get(target - num);

				if (res[0] < res[1])
					break;
			}
		}
		return res;
	}

	public ArrayList<ArrayList<Integer>> threeSum(int[] numbers) {
		// write your code here
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();

		if (numbers.length < 3)
			return res;
		Arrays.sort(numbers);

		for (int i = 0; i < numbers.length - 2; i++) {
			int j = i + 1;
			int k = numbers.length - 1;

			while (j < k) {
				int sum = numbers[i] + numbers[j] + numbers[k];
				if (sum == 0) {
					ArrayList<Integer> sol = new ArrayList<Integer>();
					sol.add(numbers[i]);
					sol.add(numbers[j]);
					sol.add(numbers[k]);
					res.add(sol);

					while (j < k && numbers[j] == numbers[j + 1])
						j++;
					while (j < k && numbers[k] == numbers[k - 1])
						k--;
					j++;
					k--;
				} else if (sum > 0)
					k--;
				else
					j++;
			}
			while (i < numbers.length - 2 && numbers[i] == numbers[i + 1])
				i++;
		}
		return res;
	}

	public int threeSumClosest(int[] numbers, int target) {
		// write your code here
		Arrays.sort(numbers);
		int res = Integer.MAX_VALUE;
		int minDif = Integer.MAX_VALUE;
		for (int i = 0; i < numbers.length - 2; i++) {
			int j = i + 1;
			int k = numbers.length - 1;

			while (j < k) {
				int sum = numbers[i] + numbers[j] + numbers[k];
				int dif = Math.abs(sum - target);

				if (dif < minDif) {
					minDif = dif;
					res = sum;
				}
				if (dif == 0)
					return target;
				else if (sum > target)
					k--;
				else
					j++;
			}
		}
		return res;
	}

	public ArrayList<ArrayList<Integer>> fourSum(int[] numbers, int target) {
		// write your code here
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		if (numbers.length < 4)
			return res;
		Arrays.sort(numbers);

		for (int i = 0; i < numbers.length - 3; i++) {
			for (int j = i + 1; j < numbers.length - 2; j++) {
				int beg = j + 1;
				int end = numbers.length - 1;
				while (beg < end) {
					int sum = numbers[i] + numbers[j] + numbers[beg]
							+ numbers[end];
					if (sum == target) {
						ArrayList<Integer> sol = new ArrayList<Integer>();
						sol.add(numbers[i]);
						sol.add(numbers[j]);
						sol.add(numbers[beg]);
						sol.add(numbers[end]);
						res.add(sol);

						while (beg < end && numbers[beg] == numbers[beg + 1])
							beg++;
						beg++;
						while (beg < end && numbers[end] == numbers[end - 1])
							end--;
						end--;
					} else if (sum > target)
						end--;
					else
						beg++;
				}
				while (j < numbers.length - 2 && numbers[j] == numbers[j + 1])
					j++;
			}
			while (i < numbers.length - 3 && numbers[i] == numbers[i + 1])
				i++;
		}
		return res;
	}

	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode A, TreeNode B) {
		// write your code here
		if (root == null || A == null || B == null)
			return null;
		if (root == A || root == B)
			return root;
		TreeNode left = lowestCommonAncestor(root.left, A, B);
		TreeNode right = lowestCommonAncestor(root.right, A, B);
		if (left != null && right != null)
			return root;
		return left == null ? right : left;
	}

	public boolean hasCycle(ListNode head) {
		// write your code here
		if (head == null || head.next == null)
			return false;
		ListNode fast = head;
		ListNode slow = head;

		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (slow == fast)
				break;
		}
		if (fast == null || fast.next == null)
			return false;
		return true;
	}

	public ListNode detectCycle(ListNode head) {
		// write your code here
		if (head == null || head.next == null)
			return null;
		ListNode fast = head;
		ListNode slow = head;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (slow == fast)
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

	public boolean isBalanced(TreeNode root) {
		// write your code here
		if (root == null)
			return true;
		int left = getHeight(root.left);
		int right = getHeight(root.right);
		if (Math.abs(left - right) > 1)
			return false;
		return isBalanced(root.left) && isBalanced(root.right);
	}

	public int getHeight(TreeNode root) {
		if (root == null)
			return 0;
		int left = getHeight(root.left);
		int right = getHeight(root.right);
		return left > right ? left + 1 : right + 1;
	}

	public List<String> anagrams(String[] strs) {
		// write your code here
		List<String> res = new ArrayList<String>();
		HashMap<String, List<String>> map = new HashMap<String, List<String>>();
		for (int i = 0; i < strs.length; i++) {
			char[] str = strs[i].toCharArray();
			Arrays.sort(str);
			String s = new String(str);
			if (map.containsKey(s))
				map.get(s).add(strs[i]);
			else {
				List<String> list = new ArrayList<String>();
				list.add(strs[i]);
				map.put(s, list);
			}
		}

		Iterator<String> it = map.keySet().iterator();

		while (it.hasNext()) {
			String s = it.next();
			List<String> lst = map.get(s);
			if (lst.size() > 1)
				res.addAll(lst);
		}
		return res;
	}

	public ArrayList<Integer> subarraySum(int[] nums) {
		// write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();

		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		int sum = 0;
		map.put(0, -1);
		for (int i = 0; i < nums.length; i++) {
			sum += nums[i];
			// if(sum==0){
			// res.add(0);
			// res.add(i);
			// break;
			// }
			if (map.containsKey(sum)) {
				res.add(map.get(sum) + 1);
				res.add(i);
				break;
			} else
				map.put(sum, i);
		}
		return res;
	}

	class Pair {
		int val;
		int index;

		public Pair(int val, int index) {
			this.val = val;
			this.index = index;
		}
	}

	public ArrayList<Integer> subarraySumClosest(int[] nums) {
		// write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();
		if (nums.length == 0)
			return res;
		Pair[] map = new Pair[nums.length];
		int minDif = Integer.MAX_VALUE;

		int sum = 0;
		for (int i = 0; i < nums.length; i++) {
			sum += nums[i];
			map[i] = new Pair(sum, i);
		}

		Arrays.sort(map, new Comparator<Pair>() {
			@Override
			public int compare(Pair p1, Pair p2) {
				return p1.val - p2.val;
			}
		});

		int beg = 0;
		int end = 0;
		for (int i = 0; i < map.length - 1; i++) {
			int diff = map[i + 1].val - map[i].val;
			if (Math.abs(diff) < minDif) {
				minDif = Math.abs(diff);
				beg = Math.min(map[i].index, map[i + 1].index) + 1;
				end = Math.max(map[i].index, map[i + 1].index);
			}
		}
		res.add(beg);
		res.add(end);
		return res;
	}

	public ArrayList<Integer> subarraySumNLogn(int[] nums) {
		ArrayList<Integer> res = new ArrayList<Integer>();
		if (nums.length == 0)
			return res;
		Pair[] map = new Pair[nums.length + 1];
		map[0] = new Pair(0, -1);
		int sum = 0;
		for (int i = 0; i < nums.length; i++) {
			sum += nums[i];
			map[i + 1] = new Pair(sum, i);
		}

		Arrays.sort(map, new Comparator<Pair>() {
			@Override
			public int compare(Pair p1, Pair p2) {
				return p1.val - p2.val;
			}
		});

		for (int i = 0; i < map.length - 1; i++) {
			if (map[i + 1].val == map[i].val) {
				int beg = Math.min(map[i].index, map[i + 1].index) + 1;
				int end = Math.max(map[i].index, map[i + 1].index);
				res.add(beg);
				res.add(end);
				return res;
			}
		}
		return res;
	}

	public int maxProfit(int[] prices) {
		// write your code here
		if (prices.length < 2)
			return 0;
		int max = 0;
		int lowest = prices[0];
		for (int i = 1; i < prices.length; i++) {
			if (prices[i] < lowest)
				lowest = prices[i];
			if (prices[i] - lowest > max)
				max = prices[i] - lowest;
		}
		return max;
	}

	public int kSum(int A[], int k, int target) {
		// write your code here
		int[] count = { 0 };
		kSumUtil(0, A, k, target, 0, 0, count);
		return count[0];
	}

	public void kSumUtil(int dep, int[] A, int k, int target, int cursum,
			int cur, int[] count) {
		if (dep == A.length || cursum > target)
			return;
		if (dep == k && cursum == target) {
			count[0]++;
			return;
		}

		for (int i = cur; i < A.length; i++) {
			cursum += A[i];
			kSumUtil(dep + 1, A, k, target, cursum, i + 1, count);
			cursum -= A[i];
		}
	}

	public int backPack(int m, int[] A) {
		// write your code here
		boolean[] canfill = new boolean[m + 1];
		canfill[0] = true;
		for (int i = 0; i < A.length; i++) {
			for (int j = m; j >= A[i]; j--) {
				canfill[j] = canfill[j] || canfill[j - A[i]];
			}
		}

		for (int i = m; i >= 0; i--) {
			if (canfill[i])
				return i;
		}
		return 0;
	}

	public int backPackII(int m, int[] A, int V[]) {
		// write your code here
		int n = A.length;
		int[][] dp = new int[n + 1][m + 1];

		for (int i = 1; i <= n; i++) {
			for (int j = 0; j <= m; j++) {
				dp[i][j] = dp[i - 1][j];
				if (j >= A[i - 1])
					dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - A[i - 1]]
							+ V[i - 1]);
			}
		}
		return dp[n][m];
	}

	/*
	 * DP map[i][j][t] denotes the number of ways to select, from first i
	 * elements, j elements whose sum equals to target
	 */

	public int kSum1(int A[], int k, int target) {
		int n = A.length;
		int[][][] dp = new int[n + 1][k + 1][target + 1];

		for (int i = 0; i <= n; i++) {
			for (int j = 0; j <= k; j++) {
				for (int t = 0; t <= target; t++) {
					if (j == 0 && t == 0)
						dp[i][j][t] = 1;
					else if (i > 0 && j > 0) {
						dp[i][j][t] = dp[i - 1][j][t];
						if (t >= A[i - 1])
							dp[i][j][t] += dp[i - 1][j - 1][t - A[i - 1]];
					}
				}
			}
		}
		return dp[n][k][target];
	}

	// 2 dimension
	// D[i][j]: k = i, target j, the solution.
	public int kSum2(int A[], int k, int target) {
		int[][] dp = new int[k + 1][target + 1];
		dp[0][0] = 1;

		for (int i = 1; i <= A.length; i++) {
			for (int j = Math.min(i, k); j >= 1; j--) {
				for (int t = target; t >= A[i - 1]; t--) {
					if (j == 1)
						dp[1][t] += A[i - 1] == t ? 1 : 0;
					else
						dp[j][t] += dp[j - 1][t - A[i - 1]];
				}
			}
		}
		return dp[k][target];
	}

	public int maxProfit3(int[] prices) {
		// write your code here
		int n = prices.length;
		if (n < 2)
			return 0;
		int[] left = new int[n];
		int lowest = prices[0];
		for (int i = 1; i < n; i++) {
			if (prices[i] < lowest)
				lowest = prices[i];
			left[i] = Math.max(left[i - 1], prices[i] - lowest);
		}

		int[] right = new int[n];
		int highest = prices[n - 1];
		for (int i = n - 2; i >= 0; i--) {
			if (prices[i] > highest)
				highest = prices[i];
			right[i] = Math.max(right[i + 1], highest - prices[i]);
		}

		int max = 0;
		for (int i = 0; i < n; i++) {
			max = Math.max(max, left[i] + right[i]);
		}
		return max;
	}

	public int binarySearch(int[] nums, int target) {
		// write your code here
		int i = 0;
		int j = nums.length - 1;
		int res = -1;
		while (i <= j) {
			int mid = (i + j) / 2;
			if (nums[mid] == target) {
				res = mid;
				j = mid - 1;
			} else if (nums[mid] > target)
				j = mid - 1;
			else
				i = mid + 1;
		}
		return res;
	}

	public ArrayList<Integer> inorderTraversal(TreeNode root) {
		// write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();
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

	public ArrayList<ArrayList<Integer>> levelOrderButtom(TreeNode root) {
		// write your code here
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		if (root == null)
			return res;
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		int curlevel = 0;
		int nextlevel = 0;
		que.add(root);
		curlevel++;
		ArrayList<Integer> level = new ArrayList<Integer>();
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
				res.add(level);
				level = new ArrayList<Integer>();
				curlevel = nextlevel;
				nextlevel = 0;
			}
		}
		Collections.reverse(res);
		return res;
	}

	public int maxPathSum(TreeNode root) {
		// write your code here
		int[] res = { Integer.MIN_VALUE };
		maxPathSum(root, res);
		return res[0];
	}

	// return single path from root
	public int maxPathSum(TreeNode root, int[] res) {
		if (root == null)
			return 0;
		int left = maxPathSum(root.left, res);
		int right = maxPathSum(root.right, res);
		int arc = left + root.val + right;

		int single = Math.max(root.val, Math.max(left, right) + root.val);

		res[0] = Math.max(res[0], Math.max(arc, single));
		return single;
	}

	public ArrayList<Integer> postorderTraversal(TreeNode root) {
		// write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();
		if (root == null)
			return res;
		Stack<TreeNode> stk = new Stack<TreeNode>();
		while (root != null) {
			stk.push(root);
			root = root.left;
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
				top = stk.pop();
				res.add(top.val);
				pre = top;
			}
		}
		return res;
	}

	public ArrayList<Integer> preorderTraversal(TreeNode root) {
		// write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();
		if (root == null)
			return res;
		Stack<TreeNode> stk = new Stack<TreeNode>();
		stk.push(root);

		while (!stk.isEmpty()) {
			TreeNode top = stk.pop();
			res.add(top.val);
			if (top.right != null)
				stk.push(top.right);
			if (top.left != null)
				stk.push(top.left);
		}
		return res;
	}

	public int climbStairs(int n) {
		// write your code here
		if (n < 2)
			return n;
		int a = 1;
		int b = 1;
		int c = 0;

		for (int i = 2; i <= n; i++) {
			c = a + b;
			a = b;
			b = c;
		}
		return b;
	}

	public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
		// write your code here
		if (node == null)
			return null;
		HashMap<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<UndirectedGraphNode, UndirectedGraphNode>();
		UndirectedGraphNode clone = new UndirectedGraphNode(node.label);

		map.put(node, clone);
		Queue<UndirectedGraphNode> que = new LinkedList<UndirectedGraphNode>();
		que.add(node);

		while (!que.isEmpty()) {
			UndirectedGraphNode top = que.remove();
			ArrayList<UndirectedGraphNode> neighbors = top.neighbors;

			for (UndirectedGraphNode n : neighbors) {
				if (!map.containsKey(n)) {
					UndirectedGraphNode copy = new UndirectedGraphNode(n.label);
					que.add(n);
					map.put(n, copy);
					map.get(top).neighbors.add(copy);
				} else
					map.get(top).neighbors.add(map.get(n));

			}
		}
		return clone;
	}

	public int maxSubArray(ArrayList<Integer> nums) {
		// write your code
		int max = 0;
		int sum = 0;
		for (int i = 0; i < nums.size(); i++) {
			sum += nums.get(i);
			if (sum < 0)
				sum = 0;
			if (sum > max)
				max = sum;
		}

		if (max == 0) {
			max = nums.get(0);
			for (int i = 1; i < nums.size(); i++) {
				max = Math.max(max, nums.get(i));
			}
		}
		return max;
	}

	public void mergeSortedArray(int[] A, int m, int[] B, int n) {
		// write your code here
		int k = m + n - 1;
		int i = m - 1;
		int j = n - 1;

		while (i >= 0 && j >= 0) {
			if (A[i] > B[j])
				A[k--] = A[i--];
			else
				A[k--] = B[j--];
		}
		while (j >= 0)
			A[k--] = B[j--];
	}

	public List<List<Integer>> combinationSum(int[] candidates, int target) {
		// write your code here
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		Arrays.sort(candidates);
		combinationSum(candidates, 0, 0, target, res, sol);
		return res;
	}

	public void combinationSum(int[] cand, int cur, int cursum, int target,
			List<List<Integer>> res, List<Integer> sol) {
		if (cur == cand.length || cursum > target)
			return;
		if (cursum == target) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}

		for (int i = cur; i < cand.length; i++) {
			cursum += cand[i];
			sol.add(cand[i]);
			combinationSum(cand, i, cursum, target, res, sol);
			cursum -= cand[i];
			sol.remove(sol.size() - 1);
		}
	}

	public List<List<Integer>> combinationSum2(int[] num, int target) {
		// write your code here
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		Arrays.sort(num);
		boolean[] visited = new boolean[num.length];
		combinationSum2Util(num, 0, 0, target, visited, sol, res);
		return res;
	}

	public void combinationSum2Util(int[] num, int cur, int cursum, int target,
			boolean[] visited, List<Integer> sol, List<List<Integer>> res) {
		if (cur > num.length || cursum > target)
			return;
		if (cursum == target) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}

		for (int i = cur; i < num.length; i++) {
			if (i != 0 && num[i] == num[i - 1] && !visited[i - 1])
				continue;
			cursum += num[i];
			sol.add(num[i]);
			visited[i] = true;
			combinationSum2Util(num, i + 1, cursum, target, visited, sol, res);
			cursum -= num[i];
			sol.remove(sol.size() - 1);
			visited[i] = false;
		}
	}

	public List<List<Integer>> combine(int n, int k) {
		// write your code here
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();

		combineUtil(0, n, k, sol, res, 1);
		return res;
	}

	public void combineUtil(int dep, int n, int k, List<Integer> sol,
			List<List<Integer>> res, int cur) {
		if (dep == k) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
			return;
		}

		for (int i = cur; i <= n; i++) {
			sol.add(i);
			combineUtil(dep + 1, n, k, sol, res, i + 1);
			sol.remove(sol.size() - 1);
		}
	}

	public boolean compareStrings(String A, String B) {
		// write your code here
		int[] hash = new int[256];
		for (int i = 0; i < B.length(); i++)
			hash[B.charAt(i)]++;
		int total = B.length();
		for (int i = 0; i < A.length(); i++) {
			char c = A.charAt(i);
			if (hash[c] == 0)
				continue;
			if (--hash[c] >= 0)
				total--;
		}
		return total == 0;
	}

	public boolean wordSegmentation(String s, Set<String> dict) {
		// write your code here
		if (s.length() == 0)
			return true;
		for (int i = 1; i <= s.length(); i++) {
			String sub = s.substring(0, i);
			if (dict.contains(sub) && wordSegmentation(s.substring(i), dict))
				return true;
		}
		return false;
	}

	public boolean wordSegmentation2(String s, Set<String> dict) {
		// write your code here
		int n = s.length();
		boolean[] dp = new boolean[n + 1];

		dp[0] = true;
		for (int i = 1; i <= n; i++) {
			for (int j = 0; j < i; j++) {
				String sub = s.substring(j, i);
				if (dp[j] && dict.contains(sub) && !dp[i])
					dp[i] = true;
			}
		}
		return dp[n];
	}

	public boolean wordSegmentation3(String s, Set<String> dict) {
		// write your code here
		int n = s.length();

		int[] chars = new int[256];
		for (String word : dict) {
			for (int i = 0; i < word.length(); i++)
				chars[word.charAt(i)]++;
		}

		for (int i = 0; i < s.length(); i++) {
			if (chars[s.charAt(i)] == 0)
				return false;
		}
		boolean[] dp = new boolean[n + 1];
		dp[0] = true;
		for (int i = 1; i <= n; i++) {
			for (int j = i - 1; j >= 0; j--) {
				if (dict.contains(s.substring(j, i)) && dp[j]) {
					dp[i] = true;
					break;
				}

			}
		}
		return dp[n];
	}

	public TreeNode buildTree(int[] inorder, int[] postorder) {
		// write your code here
		return buildTree(inorder, 0, inorder.length - 1, postorder, 0,
				postorder.length - 1);
	}

	public TreeNode buildTree(int[] inorder, int beg1, int end1,
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

		int length = index - beg1;

		root.left = buildTree(inorder, beg1, index - 1, postorder, beg2, beg2
				+ length - 1);
		root.right = buildTree(inorder, index + 1, end1, postorder, beg2
				+ length, end2 - 1);
		return root;
	}

	public TreeNode buildTree2(int[] preorder, int[] inorder) {
		// write your code here
		return buildTree2(preorder, 0, preorder.length - 1, inorder, 0,
				inorder.length - 1);
	}

	public TreeNode buildTree2(int[] preorder, int beg1, int end1,
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
		root.left = buildTree2(preorder, beg1 + 1, beg1 + length, inorder,
				beg2, index - 1);
		root.right = buildTree2(preorder, beg1 + length + 1, end1, inorder,
				index + 1, end2);
		return root;
	}

	public TreeNode sortedListToBST(ListNode head) {
		// write your code here
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

	public TreeNode sortedListToBST(ListNode head, int beg, int end) {
		if (head == null || beg > end)
			return null;
		ListNode cur = head;
		for (int i = 0; i < (end - beg) / 2; i++)
			cur = cur.next;

		TreeNode root = new TreeNode(cur.val);
		root.left = sortedListToBST(head, beg, (end + beg) / 2 - 1);
		root.right = sortedListToBST(cur.next, (end + beg) / 2 + 1, end);
		return root;
	}

	public String DeleteDigits(String A, int k) {
		// write your code here
		Stack<Integer> stk = new Stack<Integer>();
		String s = "";
		int pcount = 0;
		for (int i = 0; i < A.length(); i++) {
			int dig = A.charAt(i) - '0';
			if (stk.isEmpty() || dig >= stk.peek())
				stk.push(dig);
			else {
				if (pcount < k) {
					stk.pop();
					pcount++;
					i--;
				} else
					stk.push(dig);
			}
		}
		while (pcount < k) {
			stk.pop();
			pcount++;
		}

		while (!stk.isEmpty()) {
			s = stk.pop() + s;
		}
		int i = 0;
		while (i < s.length() && s.charAt(i) == '0')
			i++;
		return s.substring(i);
	}

	public RandomListNode copyRandomList(RandomListNode head) {
		// write your code here
		if (head == null)
			return null;
		RandomListNode cur = head;

		while (cur != null) {
			RandomListNode next = cur.next;
			RandomListNode node = new RandomListNode(cur.label);
			cur.next = node;
			node.next = next;
			cur = next;
		}

		cur = head;

		while (cur != null) {
			if (cur.random != null)
				cur.next.random = cur.random.next;
			cur = cur.next.next;
		}

		cur = head;
		RandomListNode copy = head.next;

		while (cur != null) {
			RandomListNode pnext = cur.next;
			cur.next = cur.next.next;
			cur = cur.next;
			if (cur != null)
				pnext.next = cur.next;
			pnext = pnext.next;
		}
		return copy;
	}

	public TreeNode insertNode(TreeNode root, TreeNode node) {
		// write your code here
		if (root == null) {
			root = node;
			return root;
		}

		TreeNode cur = root;
		TreeNode pre = null;
		while (cur != null) {
			if (cur.val > node.val) {
				pre = cur;
				cur = cur.left;
			} else if (cur.val < node.val) {
				pre = cur;
				cur = cur.right;
			} else
				return root;
		}
		if (node.val < pre.val)
			pre.left = node;
		else
			pre.right = node;
		return root;
	}

	public ArrayList<Interval> insert(ArrayList<Interval> intervals,
			Interval newInterval) {
		ArrayList<Interval> result = new ArrayList<Interval>();
		// write your code here
		if (intervals.size() == 0) {
			intervals.add(newInterval);
			return intervals;
		}

		boolean inserted = false;

		for (int i = 0; i < intervals.size(); i++) {
			Interval interval = intervals.get(i);
			if (interval.start < newInterval.start)
				insert(interval, result);
			else {
				insert(newInterval, result);
				insert(interval, result);
				inserted = true;
			}
		}
		if (!inserted)
			insert(newInterval, result);
		return result;
	}

	public void insert(Interval interval, ArrayList<Interval> res) {
		if (res.size() == 0) {
			res.add(interval);
			return;
		}
		Interval last = res.get(res.size() - 1);
		if (last.end < interval.start)
			res.add(interval);
		else
			last.end = Math.max(last.end, interval.end);
	}

	public int minDistance(String word1, String word2) {
		// write your code here
		int m = word1.length();
		int n = word2.length();

		int[][] dp = new int[m + 1][n + 1];

		for (int i = 0; i <= m; i++)
			dp[i][0] = i;
		for (int i = 1; i <= n; i++)
			dp[0][i] = i;

		for (int i = 1; i <= m; i++) {
			for (int j = 1; j <= n; j++) {
				if (word1.charAt(i - 1) == word2.charAt(j - 1))
					dp[i][j] = dp[i - 1][j - 1];
				else
					dp[i][j] = Math.min(Math.min(dp[i - 1][j], dp[i][j - 1]),
							dp[i - 1][j - 1]) + 1;
			}
		}
		return dp[m][n];
	}

	public int findFirstBadVersion(int n) {
		// write your code here
		int i = 1;
		int j = n;
		while (i <= j) {
			int mid = (i + j) / 2;
			if (VersionControl.isBadVersion(mid))
				j = mid - 1;
			else
				i = mid + 1;
		}
		return i;
	}

	public boolean canJump(int[] A) {
		int n = A.length;
		int maxIndex = 0;
		for (int i = 0; i < A.length; i++) {
			if (i > maxIndex || maxIndex >= n - 1)
				break;
			maxIndex = Math.max(maxIndex, i + A[i]);
		}
		return maxIndex >= n - 1 ? true : false;
	}

	public ArrayList<String> fizzBuzz(int n) {
		ArrayList<String> results = new ArrayList<String>();
		for (int i = 1; i <= n; i++) {
			if (i % 15 == 0) {
				results.add("fizz buzz");
			} else if (i % 5 == 0) {
				results.add("buzz");
			} else if (i % 3 == 0) {
				results.add("fizz");
			} else {
				results.add(String.valueOf(i));
			}
		}
		return results;
	}

	public int[] rerange(int[] A) {
		// write your code here
		int i = -1;
		int ncount = 0;// negative counts
		for (int j = 0; j < A.length; j++) {
			if (A[j] < 0) {
				ncount++;
				i++;
				swap(A, i, j);
			}
		}
		int last = i + 1;
		int first = 1;
		if (ncount <= A.length / 2) {// filp the array, longer part at the
										// first, shorter part at the second
										// half
			reverseArray(A);
			last = A.length - i - 1;// update shorter index
		}

		while (last < A.length && first < last) {
			swap(A, first, last);
			last++;
			first += 2;
		}
		return A;
	}

	public void reverseArray(int[] A) {
		int i = 0;
		int j = A.length - 1;
		while (i < j) {
			int t = A[i];
			A[i] = A[j];
			A[j] = t;
			i++;
			j--;
		}
	}

	public void swap(int[] A, int i, int j) {
		int t = A[i];
		A[i] = A[j];
		A[j] = t;
	}

	public String largestNumber(int[] num) {
		// write your code here
		String[] nums = new String[num.length];
		for (int i = 0; i < num.length; i++) {
			nums[i] = "" + num[i];
		}

		Arrays.sort(nums, new Comparator<String>() {
			@Override
			public int compare(String s1, String s2) {
				String s12 = s1 + s2;
				String s21 = s2 + s2;
				return s12.compareTo(s21);
			}

		});

		String res = "";

		for (int i = num.length - 1; i >= 0; i--) {
			res += nums[i];
		}
		int i = 0;
		while (i < res.length() && res.charAt(i) == '0')
			i++;
		if (i == res.length())
			return "0";
		return res.substring(i);
	}

	public int numDistinct(String S, String T) {
		// write your code here
		int n1 = S.length();
		int n2 = T.length();
		int[][] dp = new int[n1 + 1][n2 + 1];

		for (int i = 0; i <= n1; i++)
			dp[i][0] = 1;

		for (int i = 1; i <= n1; i++) {
			for (int j = 1; j <= n2; j++) {
				if (S.charAt(i - 1) == T.charAt(j - 1))
					dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
				else
					dp[i][j] = dp[i - 1][j];
			}
		}
		return dp[n1][n2];
	}

	public int findPeak(int[] A) {
		// write your code here
		int i = 0;
		int j = A.length - 1;
		int mid = 0;
		while (i <= j) {
			mid = (i + j) / 2;
			if ((mid == 0 || A[mid] > A[mid - 1])
					&& (mid == A.length - 1 || A[mid] > A[mid + 1]))
				return mid;
			else if (mid >= 1 && A[mid - 1] > A[mid])
				j = mid - 1;
			else
				i = mid + 1;
		}
		return mid;
	}

	public int minPathSum(int[][] grid) {
		// write your code here
		int m = grid.length;
		if (m == 0)
			return 0;
		int n = grid[0].length;
		int[][] dp = new int[m][n];
		dp[0][0] = grid[0][0];
		for (int i = 1; i < m; i++)
			dp[i][0] = dp[i - 1][0] + grid[i][0];
		for (int i = 1; i < n; i++)
			dp[0][i] = dp[0][i - 1] + grid[0][i];

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				dp[i][j] = Math.min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
			}
		}
		return dp[m - 1][n - 1];
	}

	public void heapify(int[] A) {
		// write your code here
		for (int i = A.length / 2; i >= 0; i--)
			heapify(A, i);
	}

	public void heapify(int[] A, int i) {
		if (i > A.length)
			return;
		int left = 2 * i + 1 > A.length - 1 ? Integer.MAX_VALUE : A[2 * i + 1];
		int right = 2 * i + 2 > A.length - 1 ? Integer.MAX_VALUE : A[2 * i + 2];

		if (left < right && left < A[i]) {
			A[2 * i + 1] = A[i];
			A[i] = left;
			heapify(A, 2 * i + 1);
		} else if (right <= left && right < A[i]) {
			A[2 * i + 2] = A[i];
			A[i] = right;
			heapify(A, 2 * i + 2);
		}
	}

	public void heapify2(int[] A) {
		int n = A.length;

		int siftKey = A[0];
		int keyIndex = 0;
		int left = 1;
		boolean done = false;
		while (left < n && !done) {
			int maxIndex = left;
			int right = left + 1;
			if (right < n) {
				if (A[left] < A[right])
					maxIndex = right;
			}

			if (A[maxIndex] < siftKey) {
				A[keyIndex] = A[maxIndex];
				keyIndex = maxIndex;
				left = keyIndex * 2;
			} else
				done = true;
		}
		A[keyIndex] = siftKey;
	}

	public int maxTwoSubArrays(int[] nums) {
		// write your code
		int n = nums.length;
		if (n < 2)
			return 0;
		int[] left = new int[n];
		int[] right = new int[n];

		int sum = 0;
		int max = 0;
		left[0] = nums[0];
		for (int i = 1; i < n; i++) {
			sum += nums[i - 1];
			if (sum > max) {
				max = sum;
			}
			if (sum < 0)
				sum = 0;
			left[i] = max;
		}

		sum = 0;
		max = 0;
		right[n - 1] = 0;
		for (int i = n - 2; i >= 0; i--) {
			sum += nums[i + 1];
			if (sum > max)
				max = sum;
			if (sum < 0)
				sum = 0;
			right[i] = max;
		}
		for (int i = 0; i < n; i++) {
			max = Math.max(max, left[i] + right[i]);
		}
		System.out.println("max " + max);
		if (max == 0) {
			System.out.println("ss");
			int first = nums[0];
			int second = nums[1];
			for (int i = 2; i < n; i++) {
				if (nums[i] > first) {
					second = first;
					first = nums[i];
				} else if (nums[i] > second)
					second = nums[i];
			}
			max = first + second;
			System.out.println(max);
		}
		return max;
	}

	public int maxTwoSubArrays2(ArrayList<Integer> nums) {
		int n = nums.size();
		int[] left = new int[n];
		int[] right = new int[n];

		left[0] = nums.get(0);
		int pre = nums.get(0);
		for (int i = 1; i < n; i++) {
			int sum = nums.get(i) + (pre > 0 ? pre : 0);
			pre = sum;
			left[i] = Math.max(sum, left[i - 1]);
		}

		right[n - 1] = nums.get(n - 1);
		pre = nums.get(n - 1);
		for (int i = n - 2; i >= 0; i--) {
			int sum = nums.get(i) + (pre > 0 ? pre : 0);
			pre = sum;
			right[i] = Math.max(sum, right[i + 1]);
		}
		int max = Integer.MIN_VALUE;
		for (int i = 0; i < n - 1; i++) {
			max = Math.max(max, left[i] + right[i + 1]);
		}
		return max;
	}

	/**
	 * @param nums
	 *            : A list of integers
	 * @param k
	 *            : An integer denote to find k non-overlapping subarrays
	 * @return: An integer denote the sum of max k non-overlapping subarrays
	 */

	// DP. d[i][j] means the maximum sum we can get by selecting j subarrays
	// from the first i elements.
	//
	// d[i][j] = max{d[p][j-1]+maxSubArray(p+1,i)}
	//
	// we iterate p from i-1 to j-1, so we can record the max subarray we get at
	// current p,
	// this value can be used to calculate the max subarray from p-1 to i when p
	// becomes p-1.
	public int maxSubArray(ArrayList<Integer> nums, int k) {
		// write your code
		int n = nums.size();
		if (n < k)
			return 0;
		int[][] dp = new int[n + 1][k + 1];

		for (int j = 1; j <= k; j++) {
			for (int i = j; i <= n; i++) {
				dp[i][j] = Integer.MIN_VALUE;

				int max = Integer.MIN_VALUE;
				int curmax = 0;

				for (int p = i - 1; p >= j - 1; p--) {
					curmax = Math.max(nums.get(p), curmax + nums.get(p));
					max = Math.max(max, curmax);

					if (dp[i][j] < dp[p][j - 1] + max)
						dp[i][j] = dp[p][j - 1] + max;
				}

			}
		}
		return dp[n][k];
	}

	public int maxSubArray2(ArrayList<Integer> nums, int k) {
		// write your code
		int n = nums.size();
		int[][] dp = new int[n + 1][k + 1];
		for (int i = 0; i <= n; i++) {
			for (int j = 1; j <= k; j++)
				dp[i][j] = Integer.MIN_VALUE;
		}

		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= k; j++) {
				int cmax = 0;
				for (int p = i; p >= j; p--) {
					cmax = Math.max(nums.get(p - 1), cmax + nums.get(p - 1));
					dp[i][j] = Math.max(dp[i][j], dp[p - 1][j - 1] + cmax);
				}
			}
		}
		return dp[n][k];
	}

	public boolean isValidBST(TreeNode root) {
		// write your code here
		if (root == null)
			return true;
		return isValidBST(root, Long.MIN_VALUE, Long.MAX_VALUE);
	}

	public boolean isValidBST(TreeNode root, long leftmost, long rightmost) {
		if (root == null)
			return true;
		if (root.val <= leftmost || root.val >= rightmost)
			return false;
		return isValidBST(root.left, leftmost, root.val)
				&& isValidBST(root.right, root.val, rightmost);
	}

	public boolean isInterleave(String s1, String s2, String s3) {
		// write your code here
		int n1 = s1.length();
		int n2 = s2.length();
		if (n1 + n2 != s3.length())
			return false;
		boolean[][] dp = new boolean[n1 + 1][n2 + 1];
		dp[0][0] = true;
		for (int i = 1; i <= n1; i++) {
			dp[i][0] = dp[i - 1][0] && s1.charAt(i - 1) == s3.charAt(i - 1);
		}

		for (int j = 1; j <= n2; j++)
			dp[0][j] = dp[0][j - 1] && s2.charAt(j - 1) == s3.charAt(j - 1);

		for (int i = 1; i <= n1; i++) {
			for (int j = 1; j <= n2; j++) {
				dp[i][j] = dp[i - 1][j]
						&& s1.charAt(i - 1) == s3.charAt(i + j - 1)
						|| dp[i][j - 1]
						&& s2.charAt(j - 1) == s3.charAt(i + j - 1);
			}
		}
		return dp[n1][n2];
	}

	public int jump(int[] A) {
		// write your code here
		if (A.length < 2)
			return 0;
		int maxIndex = A[0];
		int step = 1;
		int min = 0;
		while (maxIndex < A.length - 1) {
			int max = maxIndex;
			for (int i = min; i <= maxIndex; i++) {
				if (max < i + A[i]) {
					max = i + A[i];
				}
			}
			min = max;
			maxIndex = max;
			step++;
		}
		return step;
	}

	public int longestCommonSubstring(String A, String B) {
		// write your code here
		int n1 = A.length();
		int n2 = B.length();

		int[][] dp = new int[n1 + 1][n2 + 1];
		int max = 0;
		for (int i = 1; i <= n1; i++) {
			for (int j = 1; j <= n2; j++) {
				if (A.charAt(i - 1) == B.charAt(j - 1)) {
					dp[i][j] = dp[i - 1][j - 1] + 1;
					max = Math.max(max, dp[i][j]);
				} else
					dp[i][j] = 0;
			}
		}
		return max;
	}

	public int longestConsecutive(int[] num) {
		// write you code here
		int max = 0;
		HashSet<Integer> set = new HashSet<Integer>();
		for (int n : num)
			set.add(n);
		for (int i = 0; i < num.length; i++) {
			int des = findDesending(set, num[i], true);
			int ase = findDesending(set, num[i] + 1, false);
			max = Math.max(max, des + ase);
		}
		return max;
	}

	public int findDesending(HashSet<Integer> set, int num, boolean des) {
		int len = 0;
		while (set.contains(num)) {
			len++;
			set.remove(num);
			if (des)
				num--;
			else
				num++;
		}
		return len;
	}

	public int longestIncreasingSubsequence(int[] nums) {
		// write your code here
		int n = nums.length;
		int[] dp = new int[n];

		for (int i = 0; i < n; i++)
			dp[i] = 1;

		for (int i = 1; i < n; i++) {
			for (int j = 0; j < i; j++) {
				if (nums[i] >= nums[j])
					dp[i] = Math.max(dp[i], dp[j] + 1);
			}
		}

		int max = 0;
		for (int i = 0; i < n; i++)
			max = Math.max(max, dp[i]);
		return max;
	}

	public int majorityNumber(ArrayList<Integer> nums) {
		// write your code
		int number = nums.get(0);
		int count = 1;
		for (int i = 1; i < nums.size(); i++) {
			if (nums.get(i) == number)
				count++;
			else {
				if (--count == 0) {
					number = nums.get(i);
					count = 1;
				}
			}
		}
		return number;
	}

	public int majorityNumber2(ArrayList<Integer> nums) {
		// write your code
		int candidate1 = 0;
		int candidate2 = 0;
		int count1 = 0;
		int count2 = 0;

		for (int num : nums) {
			if (count1 == 0)
				candidate1 = num;
			if (count2 == 0 && num != candidate1)
				candidate2 = num;
			if (num == candidate1)
				count1++;
			if (num == candidate2)
				count2++;
			if (num != candidate1 && num != candidate2) {
				count1--;
				count2--;
			}
		}

		count1 = 0;
		count2 = 0;

		for (int num : nums) {
			if (num == candidate1)
				count1++;
			if (num == candidate2)
				count2++;
		}
		return count1 > count2 ? candidate1 : candidate2;
	}

	public int majorityNumberII(ArrayList<Integer> nums) {
		// write your code
		int n1 = nums.get(0);
		int count1 = 1;
		int i = 1;
		while (i < nums.size() && nums.get(i) == n1) {
			count1++;
			i++;
		}
		int n2 = nums.get(i);
		int count2 = 0;
		while (i < nums.size() && n2 == nums.get(i)) {
			count2++;
			i++;
		}

		while (i < nums.size()) {
			int num = nums.get(i);
			if (num == n1)
				count1++;
			else if (num == n2)
				count2++;
			else {
				if (count1 == 0) {
					n1 = num;
					count1++;
				} else if (count2 == 0) {
					n2 = num;
					count2++;
				} else {
					count1--;
					count2--;
				}
			}
			i++;
		}

		count1 = count2 = 0;
		for (int num : nums) {
			if (num == n1)
				count1++;
			if (num == n2)
				count2++;
		}
		return count1 > count2 ? n1 : n2;
	}

	public boolean isUnique(String str) {
		// write your code here
		// HashSet<Character> set=new HashSet<Character>();
		// for(int i=0;i<str.length();i++){
		// char c=str.charAt(i);
		// if(set.contains(c))
		// return false;
		// set.add(c);
		// }
		// return true;
		for (int i = 0; i < str.length(); i++) {
			for (int j = i + 1; j < str.length(); j++) {
				if (str.charAt(i) == str.charAt(j))
					return false;
			}
		}
		return true;
	}

	public int fastPower(int a, int b, int n) {
		// write your code here
		if (n == 0)
			return 1 % b;
		long res = fastPower(a, b, n / 2);
		res *= res;

		if (n % 2 == 1) {
			res = ((res % b) * (a % b)) % b;
		} else
			res %= b;
		return (int) res;
	}

	public int digitCounts(int k, int n) {
		// write your code here
		int count = 0;
		for (int i = 0; i <= n; i++) {
			count += getCounts(k, i);
		}
		return count;
	}

	public int getCounts(int k, int n) {
		int count = 0;
		if (n == 0)
			return k == 0 ? 1 : 0;
		while (n > 0) {
			int dig = n % 10;
			if (dig == k)
				count++;
			n /= 10;
		}
		return count;
	}

	public long kthPrimeNumber(int k) {
		// write your code here
		Queue<Long> q1 = new LinkedList<Long>();
		Queue<Long> q2 = new LinkedList<Long>();
		Queue<Long> q3 = new LinkedList<Long>();
		q1.add(3L);
		q2.add(5L);
		q3.add(7L);
		long val = 0;
		for (int i = 0; i < k; i++) {
			val = Math.min(q1.peek(), Math.min(q2.peek(), q3.peek()));
			if (val == q1.peek()) {
				q1.poll();
				q1.add(val * 3);
				q2.add(val * 5);
				q3.add(val * 7);
			} else if (val == q2.peek()) {
				q2.poll();
				q2.add(val * 5);
				q3.add(val * 7);
			} else {
				q3.poll();
				q3.add(val * 7);
			}
			System.out.print(val + " ");
		}
		return val;
	}

	public int majorityNumberIII(ArrayList<Integer> nums, int k) {
		// write your code
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();

		for (int num : nums) {
			if (map.containsKey(num)) {
				map.put(num, map.get(num) + 1);
			} else {
				if (map.size() < k)
					map.put(num, 1);
				else {
					List<Integer> removeKeys = new ArrayList<Integer>();
					for (int key : map.keySet()) {
						int count = map.get(key);
						count--;
						if (count == 0)
							removeKeys.add(key);
						map.put(key, count);
					}
					for (int key : removeKeys) {
						map.remove(key);
						map.put(num, 1);
					}
				}
			}
		}

		for (int key : map.keySet())
			map.put(key, 0);

		int max = 0;
		int res = 0;
		for (int num : nums) {
			if (map.containsKey(num)) {
				int count = map.get(num);
				if (++count > max) {
					max = count;
					res = num;
				}
				map.put(num, count);
			}
		}
		return res;
	}

	public ListNode addLists(ListNode l1, ListNode l2) {
		// write your code here
		if (l1 == null || l2 == null)
			return l1 == null ? l2 : l1;
		int carry = 0;
		ListNode dummy = new ListNode(0);
		ListNode pre = dummy;
		while (l1 != null && l2 != null) {
			int sum = l1.val + l2.val + carry;
			carry = sum / 10;
			sum = sum % 10;
			pre.next = new ListNode(sum);
			pre = pre.next;
			l1 = l1.next;
			l2 = l2.next;
		}
		while (l1 != null) {
			int sum = l1.val + carry;
			carry = sum / 10;
			sum = sum % 10;
			pre.next = new ListNode(sum);
			pre = pre.next;
			l1 = l1.next;
		}

		while (l2 != null) {
			int sum = l2.val + carry;
			carry = sum / 10;
			sum = sum % 10;
			pre.next = new ListNode(sum);
			pre = pre.next;
			l2 = l2.next;
		}
		if (carry == 1)
			pre.next = new ListNode(1);
		return dummy.next;
	}

	public int uniquePaths(int m, int n) {
		// write your code here
		int[][] dp = new int[m][n];
		for (int i = 0; i < m; i++)
			dp[i][0] = 1;
		for (int i = 0; i < n; i++)
			dp[0][i] = 1;

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++)
				dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
		}
		return dp[m - 1][n - 1];
	}

	public int minimumTotal(ArrayList<ArrayList<Integer>> triangle) {
		// write your code here
		int m = triangle.size();
		int n = triangle.get(m - 1).size();

		int[][] dp = new int[m][n];
		dp[0][0] = triangle.get(0).get(0);
		for (int i = 1; i < m; i++) {
			dp[i][0] = dp[i - 1][0] + triangle.get(i).get(0);
		}

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < triangle.get(i).size(); j++) {
				if (j == triangle.get(i).size() - 1)
					dp[i][j] = dp[i - 1][j - 1] + triangle.get(i).get(j);
				else
					dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j])
							+ triangle.get(i).get(j);
			}
		}
		int min = Integer.MAX_VALUE;

		for (int i = 0; i < n; i++)
			min = Math.min(min, dp[m - 1][i]);
		return min;
	}

	public boolean anagram(String s, String t) {
		// write your code here
		if (s.length() != t.length())
			return false;
		int[] count = new int[256];
		for (int i = 0; i < s.length(); i++)
			count[s.charAt(i)]++;
		for (int i = 0; i < t.length(); i++) {
			if (count[t.charAt(i)] == 0)
				return false;
			count[t.charAt(i)]--;
		}

		for (int i = 0; i < 256; i++) {
			if (count[i] != 0)
				return false;
		}
		return true;
	}

	public int median(int[] nums) {
		// write your code here
		int n = nums.length;
		if (n % 2 == 0)
			return findMedian(nums, 0, n - 1, n / 2 - 1);
		else
			return findMedian(nums, 0, n - 1, n / 2);
	}

	public int findMedian(int[] nums, int left, int right, int k) {
		int pivot = left;
		int i = left + 1;
		int j = right;
		while (i <= j) {
			while (i <= j && nums[i] <= nums[pivot])
				i++;

			while (i <= j && nums[j] > nums[pivot])
				j--;
			if (i < j) {
				swap(nums, i, j);
				i++;
				j--;
			}
		}
		swap(nums, pivot, j);
		if (j == k)
			return nums[j];
		else if (j > k)
			return findMedian(nums, left, j - 1, k);
		else
			return findMedian(nums, j + 1, right, k);
	}

	public double findMedianSortedArrays(int[] A, int[] B) {
		// write your code here
		int m = A.length;
		int n = B.length;
		if ((m + n) % 2 == 0)
			return (findKth(A, 0, B, 0, (m + n) / 2) + findKth(A, 0, B, 0,
					(m + n) / 2 + 1)) / 2.0;
		else
			return findKth(A, 0, B, 0, (m + n) / 2 + 1);
	}

	public int findKth(int[] A, int start1, int[] B, int start2, int k) {
		if (start1 >= A.length)
			return B[start2 + k - 1];
		if (start2 >= B.length)
			return A[start1 + k - 1];
		if (k == 1)
			return Math.min(A[start1], B[start2]);
		int keyA = start1 + k / 2 - 1 < A.length ? A[start1 + k / 2 - 1]
				: Integer.MAX_VALUE;
		int keyB = start2 + k / 2 - 1 < B.length ? B[start2 + k / 2 - 1]
				: Integer.MAX_VALUE;

		if (keyA < keyB)
			return findKth(A, start1 + k / 2, B, start2, k - k / 2);
		else
			return findKth(A, start1, B, start2 + k / 2, k - k / 2);
	}

	public String binaryRepresentation(String n) {
		// write your code here
		int intPart = Integer.parseInt(n.substring(0, n.indexOf('.')));
		double decPart = Double.parseDouble(n.substring(n.indexOf('.')));

		String intStr = "";
		String decStr = "";
		if (intPart == 0)
			intStr += 0;
		while (intPart > 0) {
			intStr = intPart % 2 + intStr;
			intPart /= 2;
		}

		while (decPart > 0) {
			double rem = decPart * 2;
			if (rem >= 1) {
				decPart = rem - 1;
				decStr += 1;
			} else {
				decPart = rem;
				decStr += 0;
			}
			if (decStr.length() > 32)
				return "ERROR";
		}

		return decStr.length() > 0 ? intStr + "." + decStr : intStr;
	}

	public int maxDiffSubArrays(ArrayList<Integer> nums) {
		// write your code
		int n = nums.size();
		int[] leftMax = new int[n];
		int[] leftMin = new int[n];

		leftMax[0] = leftMin[0] = nums.get(0);
		int endMax = nums.get(0);
		int endMin = nums.get(0);
		for (int i = 1; i < n; i++) {
			endMax = Math.max(nums.get(i), nums.get(i) + endMax);
			leftMax[i] = Math.max(endMax, leftMax[i - 1]);

			endMin = Math.min(nums.get(i), nums.get(i) + endMin);
			leftMin[i] = Math.min(endMin, leftMin[i - 1]);
		}

		int[] rightMax = new int[n];
		int[] rightMin = new int[n];

		rightMax[n - 1] = rightMin[n - 1] = nums.get(n - 1);
		endMax = nums.get(n - 1);
		endMin = nums.get(n - 1);

		for (int i = n - 2; i >= 0; i--) {
			endMax = Math.max(nums.get(i), nums.get(i) + endMax);
			rightMax[i] = Math.max(endMax, rightMax[i + 1]);

			endMin = Math.min(nums.get(i), nums.get(i) + endMin);
			rightMin[i] = Math.min(endMin, rightMin[i + 1]);
		}

		int maxDif = Integer.MIN_VALUE;
		for (int i = 0; i < n - 1; i++) {
			maxDif = Math.max(maxDif, Math.abs(leftMin[i] - rightMax[i + 1]));
			maxDif = Math.max(maxDif, Math.abs(leftMax[i] - rightMin[i + 1]));
		}
		return maxDif;
	}

	public int[] medianII(int[] nums) {
		int n = nums.length;
		int[] res = new int[n];
		PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();
		PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(n,
				new Comparator<Integer>() {

					@Override
					public int compare(Integer o1, Integer o2) {
						// TODO Auto-generated method stub
						return o2 - o1;
					}
				});

		for (int i = 0; i < n; i++) {
			if (minHeap.size() < maxHeap.size()) {
				if (nums[i] < maxHeap.peek()) {
					minHeap.offer(maxHeap.poll());
					maxHeap.offer(nums[i]);
				} else
					minHeap.offer(nums[i]);
			} else {// maxHeap.size() == minHeap.size())
				if (!minHeap.isEmpty() && nums[i] > minHeap.peek()) {
					maxHeap.offer(minHeap.poll());
					minHeap.offer(nums[i]);
				} else
					maxHeap.offer(nums[i]);
			}
			res[i] = maxHeap.peek();
		}
		return res;
	}

	public ListNode mergeKLists(List<ListNode> lists) {
		// write your code here
		if (lists.size() == 0)
			return null;
		PriorityQueue<ListNode> que = new PriorityQueue<ListNode>(lists.size(),
				new Comparator<ListNode>() {
					@Override
					public int compare(ListNode l1, ListNode l2) {
						return l1.val - l2.val;
					}

				});

		for (int i = 0; i < lists.size(); i++) {
			if (lists.get(i) != null)
				que.offer(lists.get(i));
		}
		ListNode dummy = new ListNode(0);
		ListNode pre = dummy;

		while (!que.isEmpty()) {
			ListNode node = que.poll();
			pre.next = node;
			pre = pre.next;
			if (node.next != null)
				que.offer(node.next);
		}
		return dummy.next;
	}

	public ArrayList<Integer> mergeSortedArray(ArrayList<Integer> A,
			ArrayList<Integer> B) {
		// write your code here
		if (A.size() == 0 || B.size() == 0)
			return A.size() == 0 ? B : A;
		int m = A.size();
		int n = B.size();

		ArrayList<Integer> res = new ArrayList<Integer>();
		int i = 0, j = 0;
		while (i < m && j < n) {
			if (A.get(i) < B.get(j))
				res.add(A.get(i++));
			else
				res.add(B.get(j++));
		}
		while (i < m)
			res.add(A.get(i++));

		while (j < n)
			res.add(B.get(j++));
		return res;
	}

	public void mergeSortedArray2(int[] A, int m, int[] B, int n) {
		// write your code here
		int i = m - 1;
		int j = n - 1;
		int k = m + n - 1;

		while (i >= 0 && j >= 0) {
			if (A[i] > B[j])
				A[k--] = A[i--];
			else
				A[k--] = B[j--];
		}

		while (j >= 0)
			A[k--] = B[j--];
	}

	public int minSubArray(ArrayList<Integer> nums) {
		// write your code
		int min = Integer.MAX_VALUE;
		int sum = 0;
		for (int i = 0; i < nums.size(); i++) {
			sum = Math.min(nums.get(i), sum + nums.get(i));
			if (sum < min)
				min = sum;
		}
		return min;
	}

	public boolean checkPowerOf2(int n) {
		// write your code here
		if (n == 0 || n < 0) {
			return false;
		}

		return (n & (n - 1)) == 0;
	}

	public ArrayList<Long> productExcludeItself(ArrayList<Integer> A) {
		// write your code
		ArrayList<Long> res = new ArrayList<Long>();
		int n = A.size();
		if (n < 2)
			return res;
		long[] left = new long[n];
		long[] right = new long[n];

		left[0] = 1;
		right[n - 1] = 1;

		for (int i = 1; i < n; i++)
			left[i] = A.get(i - 1) * left[i - 1];
		for (int i = n - 2; i >= 0; i--)
			right[i] = A.get(i + 1) * right[i + 1];

		for (int i = 0; i < n; i++) {
			res.add(left[i] * right[i]);
		}
		return res;
	}

	public void sortColors(int[] nums) {
		// write your code here
		if (nums.length < 2)
			return;
		int i = 0;
		int j = nums.length - 1;
		int k = nums.length - 1;

		while (i <= j) {
			if (nums[i] == 2) {
				nums[i] = nums[k];
				nums[k--] = 2;
				if (j > k)
					j = k;
			} else if (nums[i] == 1) {
				nums[i] = nums[j];
				nums[j--] = 1;
			} else
				i++;
		}
	}

	public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		// write your code here
		if (l1 == null || l2 == null)
			return l1 == null ? l2 : l1;
		ListNode dummy = new ListNode(0);
		ListNode pre = dummy;

		while (l1 != null && l2 != null) {
			if (l1.val < l2.val) {
				pre.next = l1;
				l1 = l1.next;
			} else {
				pre.next = l2;
				l2 = l2.next;
			}
			pre = pre.next;
		}
		if (l1 != null)
			pre.next = l1;
		if (l2 != null)
			pre.next = l2;
		return dummy.next;
	}

	public int minDepth(TreeNode root) {
		// write your code here
		if (root == null)
			return 0;
		if (root.left == null)
			return minDepth(root.right) + 1;
		if (root.right == null)
			return minDepth(root.left) + 1;
		int left = minDepth(root.left);
		int right = minDepth(root.right);
		return left < right ? left + 1 : right + 1;
	}

	ListNode nthToLast(ListNode head, int n) {
		// write your code here
		if (head == null || n == 0)
			return null;
		ListNode fast = head;

		for (int i = 0; i < n; i++)
			fast = fast.next;
		ListNode slow = head;
		while (fast != null) {
			fast = fast.next;
			slow = slow.next;
		}
		return slow;
	}

	public int partitionArray(ArrayList<Integer> nums, int k) {
		// write your code here
		int i = 0;
		int j = nums.size() - 1;
		while (i <= j) {
			while (i <= j && nums.get(i) < k)
				i++;
			while (i <= j && nums.get(j) >= k)
				j--;
			if (i < j) {
				int t = nums.get(i);
				nums.set(i, nums.get(j));
				nums.set(j, t);
				i++;
				j--;
			}
		}
		return i;
	}

	public ArrayList<ArrayList<Integer>> permute(ArrayList<Integer> nums) {
		// write your code here
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> sol = new ArrayList<Integer>();
		if (nums == null)
			return res;
		int n = nums.size();
		boolean[] visited = new boolean[n];

		permuteUtil(0, nums, visited, sol, res);
		return res;
	}

	public void permuteUtil(int dep, ArrayList<Integer> nums,
			boolean[] visited, ArrayList<Integer> sol,
			ArrayList<ArrayList<Integer>> res) {
		if (dep == nums.size()) {
			ArrayList<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}
		for (int i = 0; i < nums.size(); i++) {
			if (!visited[i]) {
				sol.add(nums.get(i));
				visited[i] = true;
				permuteUtil(dep + 1, nums, visited, sol, res);
				visited[i] = false;
				sol.remove(sol.size() - 1);
			}
		}
	}

	public ListNode partition(ListNode head, int x) {
		// write your code here
		if (head == null)
			return null;
		ListNode bigHead = new ListNode(0);
		ListNode big = bigHead;
		ListNode smallHead = new ListNode(0);
		ListNode small = smallHead;

		while (head != null) {
			if (head.val < x) {
				small.next = head;
				small = small.next;
			} else {
				big.next = head;
				big = big.next;
			}
			head = head.next;
		}
		big.next = null;
		small.next = bigHead.next;
		return smallHead.next;
	}

	public void recoverRotatedSortedArray(ArrayList<Integer> nums) {
		// write your code
		int n = nums.size();
		if (nums.get(0) < nums.get(n - 1))
			return;
		int index = -1;

		for (int i = 0; i < n - 1; i++) {
			if (nums.get(i) > nums.get(i + 1)) {
				index = i + 1;
				break;
			}
		}

		reverse(nums, 0, index - 1);
		reverse(nums, index, n - 1);
		reverse(nums, 0, n - 1);
	}

	public void reverse(ArrayList<Integer> A, int i, int j) {
		while (i < j) {
			int t = A.get(i);
			A.set(i, A.get(j));
			A.set(j, t);
			i++;
			j--;
		}
	}

	public int removeDuplicates(int[] nums) {
		// write your code here
		if (nums.length < 2)
			return nums.length;
		int i = 0;
		for (int j = 1; j < nums.length; j++) {
			if (nums[j] != nums[i])
				nums[++i] = nums[j];
		}
		return ++i;
	}

	public int removeDuplicates2(int[] nums) {
		// write your code here
		if (nums.length < 3)
			return nums.length;
		int i = 0;
		int count = 1;

		for (int j = 1; j < nums.length; j++) {
			if (nums[j] != nums[i]) {
				nums[++i] = nums[j];
				count = 1;
			} else {
				if (count < 2) {
					nums[++i] = nums[j];
					count++;
				}
			}
		}
		return i + 1;
	}

	public int removeDuplicates3(int[] nums) {
		// write your code here
		if (nums.length < 3)
			return nums.length;
		int i = 1;
		int count = 1;

		for (int j = 1; j < nums.length; j++) {
			if (nums[j] == nums[j - 1])
				count++;
			else
				count = 1;
			if (count <= 2)
				nums[i++] = nums[j];
		}
		return i;
	}

	public static ListNode deleteDuplicates(ListNode head) {
		// write your code here
		if (head == null || head.next == null)
			return head;
		ListNode pre = head;
		ListNode cur = head.next;

		while (cur != null) {
			if (cur.val != pre.val) {
				pre = pre.next;
				cur = cur.next;
			} else {
				cur = cur.next;
				pre.next = cur;
			}
		}
		return head;
	}

	public ListNode deleteDuplicatesAll(ListNode head) {
		// write your code here
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(Integer.MAX_VALUE);
		dummy.next = head;
		ListNode pre = dummy;
		ListNode cur = head;

		while (cur != null) {
			boolean dup = false;
			while (cur.next != null && cur.val == cur.next.val) {
				cur = cur.next;
				dup = true;
			}
			if (dup)
				pre.next = cur.next;
			else {
				pre.next = cur;
				pre = pre.next;
			}
			cur = cur.next;
		}
		return dummy.next;
	}

	/**
	 * @param ListNode
	 *            head is the head of the linked list
	 * @return: ListNode head of the linked list
	 */
	public ListNode deleteDuplicatesAll2(ListNode head) {
		// write your code here
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;

		head = dummy;

		while (head.next != null && head.next.next != null) {
			if (head.next.val == head.next.next.val) {
				int val = head.next.val;
				while (head.next != null && head.next.val == val) {
					head.next = head.next.next;
				}
			} else
				head = head.next;
		}
		return dummy.next;
	}

	public int removeElement(int[] A, int elem) {
		// write your code here
		int j = 0;
		for (int i = 0; i < A.length; i++) {
			if (A[i] != elem) {
				A[j++] = A[i];
			}
		}
		return j;
	}

	ListNode removeNthFromEnd(ListNode head, int n) {
		// write your code here
		if (head == null)
			return null;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode slow = head;
		ListNode pre = dummy;

		ListNode fast = head;
		for (int i = 0; i < n; i++) {
			if (fast == null)
				return null;
			fast = fast.next;
		}
		while (fast != null) {
			fast = fast.next;
			pre = slow;
			slow = slow.next;
		}
		pre.next = slow.next;
		return dummy.next;

	}

	public ListNode reverse(ListNode head) {
		// write your code here
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

	public ListNode reverseBetween(ListNode head, int m, int n) {
		// write your code
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode pre = dummy;
		ListNode cur = head;
		for (int i = 0; i < m - 1; i++) {
			pre = cur;
			cur = cur.next;
		}
		ListNode before = pre;
		ListNode last = cur;
		pre = cur;
		cur = cur.next;
		for (int i = 0; i < n - m; i++) {
			ListNode pnext = cur.next;
			cur.next = pre;
			pre = cur;
			cur = pnext;
		}

		before.next = pre;
		last.next = cur;
		return dummy.next;
	}

	public String reverseWords(String s) {
		// write your code
		StringBuilder sb = new StringBuilder();

		String[] strs = s.split(" ");
		for (int i = strs.length - 1; i >= 0; i--) {
			if (!strs[i].equals("")) {
				sb.append(strs[i]);
				sb.append(" ");
			}
		}
		return sb.toString().trim();

	}

	public ListNode rotateRight(ListNode head, int k) {
		// write your code here
		if (head == null)
			return null;
		int len = 0;
		ListNode cur = head;
		while (cur != null) {
			len++;
			cur = cur.next;
		}
		k = k % len;
		if (k == 0)
			return head;
		cur = head;
		ListNode pre = null;
		for (int i = 0; i < len - k; i++) {
			pre = cur;
			cur = cur.next;
		}
		ListNode newHead = cur;
		pre.next = null;
		while (cur != null) {
			pre = cur;
			cur = cur.next;
		}
		pre.next = head;
		return newHead;
	}

	public ListNode rotateRight2(ListNode head, int k) {
		// write your code here
		if (head == null || head.next == null)
			return head;
		int len = 0;
		ListNode cur = head;
		while (cur != null) {
			len++;
			cur = cur.next;
		}
		k = k % len;
		if (k == 0)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		head = dummy;
		ListNode tail = dummy;

		for (int i = 0; i < k; i++)
			head = head.next;
		while (head.next != null) {
			tail = tail.next;
			head = head.next;
		}
		ListNode newHead = tail.next;
		head.next = dummy.next;
		tail.next = null;
		return newHead;
	}

	public ArrayList<ArrayList<Integer>> subsetsWithDup(ArrayList<Integer> S) {
		// write your code here
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		ArrayList<Integer> sol = new ArrayList<Integer>();
		if (S == null || S.size() == 0)
			return res;
		Collections.sort(S);
		subsetsWithDupUtil(0, S, sol, res);
		return res;
	}

	public void subsetsWithDupUtil(int cur, ArrayList<Integer> S,
			ArrayList<Integer> sol, ArrayList<ArrayList<Integer>> res) {
		res.add(new ArrayList<Integer>(sol));
		for (int i = cur; i < S.size(); i++) {
			if (i != cur && S.get(i) == S.get(i - 1))
				continue;
			sol.add(S.get(i));
			subsetsWithDupUtil(i + 1, S, sol, res);
			sol.remove(sol.size() - 1);
		}
	}

	public int singleNumber(int[] A) {
		if (A.length == 0) {
			return 0;
		}

		int n = A[0];
		for (int i = 1; i < A.length; i++) {
			n = n ^ A[i];
		}

		return n;
	}

	public int singleNumberII(int[] A) {
		// write your code here
		int res = 0;
		for (int i = 0; i < 32; i++) {
			int x = 1 << i;
			int sum = 0;
			for (int j = 0; j < A.length; j++) {
				if ((x & A[j]) != 0)
					sum++;
			}
			if (sum % 3 != 0)
				res |= x;
		}
		return res;
	}

	public char[] rotateString(char[] A, int offset) {
		// wirte your code here
		int n = A.length;
		if (n == 0)
			return A;
		offset %= n;

		reverseArray(A, 0, A.length - offset - 1);
		reverseArray(A, A.length - offset, A.length - 1);
		reverseArray(A, 0, A.length - 1);
		return A;
	}

	public void reverseArray(char[] A, int i, int j) {
		while (i < j) {
			char c = A[i];
			A[i] = A[j];
			A[j] = c;
			i++;
			j--;
		}
	}

	public void rotate(int[][] matrix) {
		// write your code here
		int n = matrix.length;
		for (int i = 0; i < n; i++) {
			for (int j = i; j < n; j++) {
				int t = matrix[i][j];
				matrix[i][j] = matrix[j][i];
				matrix[j][i] = t;
			}
		}

		for (int i = 0; i < n; i++) {
			int beg = 0;
			int end = n - 1;
			while (beg < end) {
				int t = matrix[i][beg];
				matrix[i][beg] = matrix[i][end];
				matrix[i][end] = t;
				beg++;
				end--;
			}
		}
	}

	public boolean searchMatrix(int[][] matrix, int target) {
		// write your code here
		int m = matrix.length;
		if (m == 0)
			return false;
		int n = matrix[0].length;

		int i = 0;
		int j = n - 1;
		while (i < m && j >= 0) {
			if (matrix[i][j] == target)
				return true;
			else if (matrix[i][j] < target)
				i++;
			else
				j--;
		}
		return false;
	}

	// binary search
	public boolean searchMatrix2(int[][] matrix, int target) {
		// write your code here
		int m = matrix.length;
		if (m == 0)
			return false;
		int n = matrix[0].length;

		if (target < matrix[0][0])
			return false;

		int start = 0;
		int end = m - 1;
		while (start <= end) {
			int mid = (start + end) / 2;
			if (matrix[mid][0] == target)
				return true;
			else if (matrix[mid][0] < target)
				start = mid + 1;
			else
				end = mid - 1;
		}
		int targetRow = end;
		start = 0;
		end = n - 1;

		while (start <= end) {
			int mid = (start + end) / 2;
			if (matrix[targetRow][mid] == target)
				return true;
			else if (matrix[targetRow][mid] < target)
				start = mid + 1;
			else
				end = mid - 1;
		}
		return false;
	}

	public int searchMatrixII(int[][] matrix, int target) {
		// write your code here
		int m = matrix.length;
		if (m == 0)
			return 0;
		int n = matrix[0].length;

		int i = 0;
		int j = n - 1;
		int count = 0;
		while (i < m && j >= 0) {
			if (matrix[i][j] == target) {
				count++;
				i++;
				j--;
			} else if (matrix[i][j] < target)
				i++;
			else
				j--;
		}
		return count;
	}

	public ArrayList<ArrayList<Integer>> permuteUnique(ArrayList<Integer> nums) {
		// write your code here
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		if (nums == null || nums.size() == 0)
			return res;
		ArrayList<Integer> sol = new ArrayList<Integer>();
		boolean[] visited = new boolean[nums.size()];
		Collections.sort(nums);
		permuteUnique(0, nums, visited, sol, res);
		return res;
	}

	public void permuteUnique(int dep, ArrayList<Integer> nums,
			boolean[] visited, ArrayList<Integer> sol,
			ArrayList<ArrayList<Integer>> res) {
		if (dep == nums.size()) {
			ArrayList<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}

		for (int i = 0; i < nums.size(); i++) {
			if (!visited[i]) {
				if (i != 0 && nums.get(i) == nums.get(i - 1) && !visited[i - 1])
					continue;
				visited[i] = true;
				sol.add(nums.get(i));
				permuteUnique(dep + 1, nums, visited, sol, res);
				visited[i] = false;
				sol.remove(sol.size() - 1);
			}
		}
	}

	public List<List<String>> partition(String s) {
		// write your code here
		List<List<String>> res = new ArrayList<List<String>>();
		if (s.length() == 0)
			return res;
		List<String> sol = new ArrayList<String>();
		partitionUtil(s, sol, res);
		return res;
	}

	public void partitionUtil(String s, List<String> sol, List<List<String>> res) {
		if (s.length() == 0) {
			res.add(new ArrayList<String>(sol));
		}

		for (int i = 0; i < s.length(); i++) {
			if (isPalindrome(s.substring(0, i + 1))) {
				sol.add(s.substring(0, i + 1));
				partitionUtil(s.substring(i + 1), sol, res);
				sol.remove(sol.size() - 1);
			}
		}
	}

	public boolean isPalindrome(String s) {
		if (s.length() < 2)
			return true;
		int i = 0;
		int j = s.length() - 1;
		while (i < j) {
			if (s.charAt(i) != s.charAt(j))
				return false;
			i++;
			j--;
		}
		return true;
	}

	public int uniquePathsWithObstacles(int[][] obstacleGrid) {
		// write your code here
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

	public boolean exist(char[][] board, String word) {
		// write your code here
		int m = board.length;
		if (m == 0)
			return false;
		int n = board[0].length;
		boolean[][] used = new boolean[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == word.charAt(0)) {
					if (findWord(board, i, j, word, used, 0))
						return true;
				}
			}
		}
		return false;
	}

	public boolean findWord(char[][] board, int i, int j, String word,
			boolean[][] used, int cur) {
		if (cur == word.length())
			return true;
		if (i < 0 || i >= board.length || j < 0 || j >= board[0].length
				|| board[i][j] != word.charAt(cur) || used[i][j])
			return false;
		if (board[i][j] == word.charAt(cur) && !used[i][j]) {
			used[i][j] = true;
			boolean exist = findWord(board, i + 1, j, word, used, cur + 1)
					|| findWord(board, i - 1, j, word, used, cur + 1)
					|| findWord(board, i, j + 1, word, used, cur + 1)
					|| findWord(board, i, j - 1, word, used, cur + 1);
			if (exist)
				return true;
			else
				used[i][j] = false;
		}
		return false;
	}

	public ArrayList<String> wordSearchII(char[][] board,
			ArrayList<String> words) {
		// write your code here
		ArrayList<String> res = new ArrayList<String>();
		int m = board.length;
		if (m == 0)
			return res;
		int n = board[0].length;
		boolean[][] used = new boolean[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				findWords(board, i, j, words, used, "", res);
			}
		}
		return res;
	}

	public void findWords(char[][] board, int i, int j,
			ArrayList<String> words, boolean[][] used, String s,
			ArrayList<String> res) {
		if (words.contains(s)) {
			if (!res.contains(s))
				res.add(s);
		}
		if (i < 0 || i >= board.length || j < 0 || j >= board[0].length
				|| used[i][j])
			return;
		if (!used[i][j]) {
			used[i][j] = true;
			findWords(board, i + 1, j, words, used, s + board[i][j], res);
			findWords(board, i - 1, j, words, used, s + board[i][j], res);
			findWords(board, i, j - 1, words, used, s + board[i][j], res);
			findWords(board, i, j + 1, words, used, s + board[i][j], res);
			used[i][j] = false;
		}
	}

	public void reorderList(ListNode head) {
		// write your code here
		if (head == null || head.next == null)
			return;
		ListNode fast = head;
		ListNode slow = head;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			if (fast == null)
				break;
			slow = slow.next;
		}

		ListNode second = slow.next;
		slow.next = null;
		second = reverseList(second);

		ListNode cur = head;

		while (head != null && second != null) {
			ListNode pnext1 = cur.next;
			cur.next = second;
			ListNode pnext2 = second.next;
			second.next = pnext1;

			cur = pnext1;
			second = pnext2;
		}
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

	public ArrayList<Integer> searchRange(ArrayList<Integer> A, int target) {
		// write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();
		res.add(-1);
		res.add(-1);

		int beg = 0;
		int end = A.size() - 1;
		int index = -1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A.get(mid) == target) {
				index = mid;
				break;
			} else if (A.get(mid) > target)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		if (index == -1)
			return res;
		beg = 0;
		end = index - 1;

		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A.get(mid) == target)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		res.set(0, beg);

		beg = index + 1;
		end = A.size() - 1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A.get(mid) == target)
				beg = mid + 1;
			else
				end = mid - 1;
		}
		res.set(1, end);
		return res;
	}

	public int search(int[] A, int target) {
		// write your code here
		if (A.length == 0)
			return -1;
		int beg = 0;
		int end = A.length - 1;

		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A[mid] == target)
				return mid;
			else if (A[beg] <= A[mid]) {
				if (A[beg] <= target && target < A[mid])
					end = mid - 1;
				else
					beg = mid + 1;
			} else {
				if (A[mid] < target && target <= A[end])
					beg = mid + 1;
				else
					end = mid - 1;
			}
		}
		return -1;
	}

	public int searchInsert(int[] A, int target) {
		// write your code here
		if (A.length == 0)
			return 0;
		int beg = 0;
		int end = A.length - 1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A[mid] == target)
				return mid;
			else if (A[mid] > target)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		return beg;
	}

	public ArrayList<Integer> searchRange(TreeNode root, int k1, int k2) {
		// write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();
		searchRange(root, k1, k2, res);
		return res;
	}

	public void searchRange(TreeNode root, int k1, int k2,
			ArrayList<Integer> res) {
		if (root == null)
			return;

		if (root.val > k1)
			searchRange(root.left, k1, k2, res);

		if (root.val >= k1 && root.val <= k2)
			res.add(root.val);

		if (root.val < k2)
			searchRange(root.right, k1, k2, res);
	}

	public ArrayList<Integer> searchRangeBST(TreeNode root, int k1, int k2) {
		ArrayList<Integer> result = new ArrayList<Integer>();
		Stack<TreeNode> stack = new Stack<TreeNode>();
		TreeNode cur = root;
		while (cur != null || !stack.isEmpty()) {
			if (cur == null) {
				cur = stack.pop();
				if (cur.val <= k2) {
					if (cur.val >= k1)
						result.add(cur.val); // 1!
					cur = cur.right;
				} else {
					cur = null; // 2!
				}
			} else {
				stack.add(cur);
				if (cur.val >= k1) {
					cur = cur.left;
				} else {
					cur = null; // 3!
				}
			}
		}
		return result;
	}

	public int numTrees(int n) {
		// write your code here
		if (n == 0 || n == 1)
			return 1;
		int total = 0;
		for (int i = 1; i <= n; i++) {
			total += numTrees(i - 1) * numTrees(n - i);
		}
		return total;
	}

	public void setZeroes(int[][] matrix) {
		// write your code here
		int m = matrix.length;
		if (m == 0)
			return;
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
				if (row[i] || col[j])
					matrix[i][j] = 0;
			}
		}
	}

	public void setZeroes2(int[][] matrix) {
		// write your code here
		int m = matrix.length;
		if (m == 0)
			return;
		int n = matrix[0].length;
		boolean fr = false;
		boolean fc = false;

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
				if (matrix[i][0] == 0 || matrix[0][j] == 0)
					matrix[i][j] = 0;
			}
		}
		if (fr) {
			for (int i = 0; i < n; i++)
				matrix[0][i] = 0;
		}
		if (fc) {
			for (int i = 0; i < m; i++)
				matrix[i][0] = 0;
		}

	}

	public void sortLetters(char[] chars) {
		// write your code here
		int i = 0;
		int j = chars.length - 1;

		while (i < j) {
			while (i < j && Character.isLowerCase(chars[i]))
				i++;

			while (i < j && Character.isUpperCase(chars[j]))
				j--;
			if (i < j) {
				char c = chars[i];
				chars[i] = chars[j];
				chars[j] = c;
				i++;
				j--;
			}
		}
	}

	public ArrayList<Integer> nextPermuation(ArrayList<Integer> nums) {
		// write your code
		if (nums.size() < 2)
			return nums;
		int index = -1;

		for (int i = 0; i < nums.size() - 1; i++) {
			if (nums.get(i) < nums.get(i + 1))
				index = i;
		}
		if (index == -1) {
			Collections.sort(nums);
			return nums;
		}

		int idx = index + 1;
		for (int i = index + 1; i < nums.size(); i++) {
			if (nums.get(i) > nums.get(index))
				idx = i;
		}

		int t = nums.get(index);
		nums.set(index, nums.get(idx));
		nums.set(idx, t);

		int beg = index + 1;
		int end = nums.size() - 1;

		while (beg < end) {
			int tmp = nums.get(beg);
			nums.set(beg, nums.get(end));
			nums.set(end, tmp);
			beg++;
			end--;
		}
		return nums;
	}

	public ArrayList<Integer> previousPermuation(ArrayList<Integer> nums) {
		// write your code
		if (nums.size() < 2)
			return nums;
		int index = -1;

		for (int i = 0; i < nums.size() - 1; i++) {
			if (nums.get(i) > nums.get(i + 1))
				index = i;
		}
		if (index == -1) {
			Collections.reverse(nums);
			return nums;
		}

		int idx = index + 1;

		for (int i = index + 1; i < nums.size(); i++) {
			if (nums.get(i) < nums.get(index))
				idx = i;
		}

		int tmp = nums.get(index);
		nums.set(index, nums.get(idx));
		nums.set(idx, tmp);

		int beg = index + 1;
		int end = nums.size() - 1;

		while (beg < end) {
			tmp = nums.get(beg);
			nums.set(beg, nums.get(end));
			nums.set(end, tmp);
			beg++;
			end--;
		}
		return nums;
	}

	public int strStr(String source, String target) {
		// write your code here
		if (target == null || source == null
				|| source.length() < target.length())
			return -1;
		if (source.length() == 0)
			return 0;
		for (int i = 0; i < source.length(); i++) {
			int j = 0;
			for (; j < target.length(); j++) {
				if (target.charAt(j) != source.charAt(i + j))
					break;
			}
			if (j == target.length())
				return i;
		}
		return -1;
	}

	public int largestRectangleArea(int[] height) {
		// write your code here
		int[] h = Arrays.copyOf(height, height.length + 1);
		Stack<Integer> stk = new Stack<Integer>();
		int max = 0;

		int i = 0;
		while (i < h.length) {
			if (stk.isEmpty() || h[i] >= h[stk.peek()]) {
				stk.push(i++);
			} else {
				int top = stk.pop();
				int length = stk.isEmpty() ? i : i - stk.peek() - 1;
				max = Math.max(max, h[top] * length);
			}
		}
		return max;
	}

	public int numTrees2(int n) {
		// write your code here
		int[] count = new int[n + 1];
		count[0] = 1;

		for (int i = 1; i <= n; i++) {
			for (int j = 0; j < i; j++)
				count[i] += count[j] * count[i - j - 1];
		}
		return count[n];
	}

	public List<TreeNode> generateTrees(int n) {
		// write your code here
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

	public String minWindow(String source, String target) {
		// write your code
		if (source == null || target == null
				|| target.length() > source.length())
			return "";
		if (source.length() == 0)
			return "";
		int toFind[] = new int[256];
		for (int i = 0; i < target.length(); i++)
			toFind[target.charAt(i)]++;
		int[] hasFound = new int[256];
		int count = target.length();
		int windowStart = 0;
		int windowEnd = 0;
		int minLen = source.length() + 1;
		int start = 0;
		for (int i = 0; i < source.length(); i++) {
			char c = source.charAt(i);
			if (toFind[c] == 0)
				continue;
			hasFound[c]++;
			if (hasFound[c] <= toFind[c])
				count--;
			if (count == 0) {
				while (toFind[source.charAt(start)] == 0
						|| hasFound[source.charAt(start)] > toFind[source
								.charAt(start)]) {
					if (hasFound[source.charAt(start)] > toFind[source
							.charAt(start)])
						hasFound[source.charAt(start)]--;
					start++;
				}
				if (i - start + 1 < minLen) {
					minLen = i - start + 1;
					windowStart = start;
					windowEnd = i;
				}
			}
		}
		if (count == 0)
			return source.substring(windowStart, windowEnd + 1);
		return "";
	}

	/**
	 * This method will be invoked first, you should design your own algorithm
	 * to serialize a binary tree which denote by a root node to a string which
	 * can be easily deserialized by your own "deserialize" method later.
	 */
	public String serialize(TreeNode root) {
		// write your code here
		String[] res = { "" };
		serializePreorderUtil(root, res);
		return res[0];
	}

	public void serializePreorderUtil(TreeNode root, String[] res) {
		if (root == null) {
			res[0] += "# ";
		} else {
			res[0] += root.val + " ";
			serializePreorderUtil(root.left, res);
			serializePreorderUtil(root.right, res);
		}
	}

	/**
	 * This method will be invoked second, the argument data is what exactly you
	 * serialized at method "serialize", that means the data is not given by
	 * system, it's given by your own serialize method. So the format of data is
	 * designed by yourself, and deserialize it here as you serialize it in
	 * "serialize" method.
	 */
	public TreeNode deserialize(String data) {
		// write your code here
		String[] tokens = data.trim().split(" ");
		int[] index = { 0 };
		return deserializeUtil(tokens, index);
	}

	public TreeNode deserializeUtil(String[] tokens, int[] index) {
		if (index[0] >= tokens.length)
			return null;
		if (tokens[index[0]].equals("#")) {
			index[0]++;
			return null;
		}

		int val = Integer.parseInt(tokens[index[0]]);
		TreeNode root = new TreeNode(val);
		index[0]++;
		root.left = deserializeUtil(tokens, index);
		root.right = deserializeUtil(tokens, index);

		return root;
	}

	public int woodCut(int[] L, int k) {
		// write your code here
		int lo = 1;
		int hi = 0;
		for (int l : L)
			hi = Math.max(hi, l);
		int max = 0;
		while (lo <= hi) {
			int mid = lo + (hi - lo) / 2;
			int count = 0;
			for (int l : L)
				count += (l / mid);
			if (count < k)
				hi = mid - 1;
			else {
				max = mid;
				lo = mid + 1;
			}
		}
		return max;
	}

	public ListNode sortList(ListNode head) {
		// write your code here
		if (head == null || head.next == null)
			return head;
		ListNode fast = head;
		ListNode slow = head;

		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			if (fast == null)
				break;
			slow = slow.next;
		}

		ListNode second = slow.next;
		slow.next = null;

		ListNode first = sortList(head);
		second = sortList(second);

		head = mergeLists(first, second);
		return head;
	}

	public ListNode mergeLists(ListNode l1, ListNode l2) {
		if (l1 == null || l2 == null)
			return l1 == null ? l2 : l1;
		ListNode dummy = new ListNode(0);
		ListNode pre = dummy;

		while (l1 != null && l2 != null) {
			if (l1.val < l2.val) {
				pre.next = l1;
				l1 = l1.next;
			} else {
				pre.next = l2;
				l2 = l2.next;
			}
			pre = pre.next;
		}
		if (l1 != null)
			pre.next = l1;
		if (l2 != null)
			pre.next = l2;
		return dummy.next;
	}

	public int minCut(String s) {
		// write your code here
		int n = s.length();
		int[][] dp = new int[n][n];
		boolean[][] p = new boolean[n][n];

		for (int i = 0; i < n; i++) {
			p[i][i] = true;
			dp[i][i] = 0;
		}

		for (int len = 2; len <= n; len++) {
			for (int i = 0; i < n - len + 1; i++) {
				int j = i + len - 1;
				if (s.charAt(i) == s.charAt(j)) {
					if (j == i + 1)
						p[i][j] = true;
					else
						p[i][j] = p[i + 1][j - 1];
				}
				if (p[i][j])
					dp[i][j] = 0;
				else {
					dp[i][j] = Integer.MAX_VALUE;
					for (int k = i; k < j; k++) {
						dp[i][j] = Math.min(dp[i][j], dp[i][k] + dp[k + 1][j]
								+ 1);
					}
				}
			}
		}
		return dp[0][n - 1];
	}

	public int minCut2(String s) {
		// write your code here
		int n = s.length();
		int[] dp = new int[n + 1];
		boolean[][] p = new boolean[n][n];
		for (int i = n; i >= 0; i--)
			dp[i] = n - i;

		for (int i = n - 1; i >= 0; i--) {
			for (int j = i; j < n; j++) {
				if (s.charAt(i) == s.charAt(j)
						&& (j - i < 2 || p[i + 1][j - 1])) {
					p[i][j] = true;
					dp[i] = Math.min(dp[i], dp[j + 1] + 1);
				}
			}
		}
		return dp[0] - 1;
	}

	public int hashCode(char[] key, int HASH_SIZE) {
		// write your code here
		long hash = 0;
		int n = key.length;
		long base = 1;
		for (int i = n - 1; i >= 0; i--) {
			if (i == n - 1)
				base = 1;
			else
				base = (base * 33) % HASH_SIZE;
			hash = (((int) key[i] * base) % HASH_SIZE + hash) % HASH_SIZE;
		}
		return (int) hash;
	}

	// needs to be done again
	public void sortColors2(int[] colors, int k) {
		// write your code here
		int[] count = new int[k];
		for (int i = 0; i < colors.length; i++)
			count[colors[i] - 1]++;
		int j = 0;
		for (int i = 0; i < k; i++) {
			while (count[i] > 0) {
				colors[j++] = i + 1;
				count[i]--;
			}
		}
	}

	public int bitSwapRequired(int a, int b) {
		// write your code here
		int xor = a ^ b;
		int result = 0;
		while (xor != 0) {// instead of xor > 0 !important
			result++;
			xor &= (xor - 1);
		}
		return result;
	}

	public TreeNode maxTree(int[] A) {
		// write your code here
		if (A.length == 0)
			return null;
		return maxTree(A, 0, A.length - 1);
	}

	public TreeNode maxTree(int[] A, int beg, int end) {
		if (beg > end)
			return null;
		int max = A[beg];
		int index = beg;
		for (int i = beg + 1; i <= end; i++) {
			if (A[i] > max) {
				max = A[i];
				index = i;
			}
		}

		TreeNode root = new TreeNode(A[index]);
		root.left = maxTree(A, beg, index - 1);
		root.right = maxTree(A, index + 1, end);
		return root;
	}

	public TreeNode maxTree2(int[] A) {
		if (A.length == 0)
			return null;
		TreeNode root = null;

		for (int num : A) {
			TreeNode node = new TreeNode(num);
			if (root == null)
				root = node;
			else if (root.val < num) {
				node.left = root;
				root = node;
			} else {
				TreeNode cur = root;
				while (cur != null) {
					if (cur.right == null) {
						cur.right = node;
						break;
					}
					if (cur.right.val < num) {
						node.left = cur.right;
						cur.right = node;
						break;
					}
					cur = cur.right;
				}
			}
		}
		return root;
	}

	public ListNode[] rehashing(ListNode[] hashTable) {
		// write your code here
		int n = hashTable.length;
		ListNode[] newhash = new ListNode[2 * n];

		for (int i = 0; i < n; i++) {
			ListNode node = hashTable[i];
			// if(node!=null){
			while (node != null) {
				int val = node.val;
				ListNode newnode = new ListNode(val);
				int index = -1;
				if (val >= 0) {
					index = val % (2 * n);
				} else {
					index = (val % (2 * n) + 2 * n) % (2 * n);
				}
				if (newhash[index] == null) {
					newhash[index] = newnode;
				} else {
					ListNode cur = newhash[index];
					while (cur.next != null)
						cur = cur.next;
					cur.next = newnode;
				}
				node = node.next;
			}
			// }
		}
		return newhash;
	}

	public ListNode[] rehashing2(ListNode[] hashTable) {
		// write your code here
		int n = hashTable.length;
		int size = 2 * n;
		ListNode[] newhash = new ListNode[size];

		for (int i = 0; i < n; i++) {
			ListNode node = hashTable[i];

			while (node != null) {
				ListNode t = node;
				node = node.next;
				int index = (t.val % size + size) % size;
				if (newhash[index] == null) {
					newhash[index] = t;
					t.next = null;
				} else {
					ListNode cur = newhash[index];
					while (cur.next != null)
						cur = cur.next;
					cur.next = t;
					t.next = null;
				}
			}
		}
		return newhash;
	}

	ArrayList<ArrayList<String>> solveNQueens(int n) {
		// write your code here
		ArrayList<ArrayList<String>> res = new ArrayList<ArrayList<String>>();
		int[] loc = new int[n];
		solveNQueens(0, n, loc, res);
		return res;
	}

	public void solveNQueens(int cur, int n, int[] loc,
			ArrayList<ArrayList<String>> res) {
		if (cur == n) {
			printBoard(loc, res);
			return;
		}
		for (int i = 0; i < n; i++) {
			loc[cur] = i;
			if (isValid(cur, loc)) {
				solveNQueens(cur + 1, n, loc, res);
			}
		}
	}

	public boolean isValid(int cur, int[] loc) {
		for (int i = 0; i < cur; i++) {
			if (loc[i] == loc[cur] || Math.abs(loc[i] - loc[cur]) == cur - i)
				return false;
		}
		return true;
	}

	public void printBoard(int[] loc, ArrayList<ArrayList<String>> res) {
		ArrayList<String> sol = new ArrayList<String>();

		for (int i = 0; i < loc.length; i++) {
			String row = "";
			for (int j = 0; j < loc.length; j++) {
				if (loc[i] == j) {
					row += 'Q';
				} else
					row += '.';
			}
			sol.add(row);
		}
		res.add(sol);
	}

	public int totalNQueens(int n) {
		// write your code here
		int[] count = { 0 };
		int[] loc = new int[n];
		totalNQueens(0, n, loc, count);
		return count[0];
	}

	public void totalNQueens(int cur, int n, int[] loc, int[] count) {
		if (cur == n) {
			count[0]++;
			return;
		}

		for (int i = 0; i < n; i++) {
			loc[cur] = i;
			if (isValid(cur, loc))
				totalNQueens(cur + 1, n, loc, count);
		}
	}

	public List<Integer> singleNumberIII(int[] A) {
		// write your code here
		List<Integer> res = new ArrayList<Integer>();
		int xor = 0;
		for (int num : A)
			xor ^= num;
		int x = 0;
		int y = 0;

		int mask = 1;
		while ((xor & mask) == 0)
			mask <<= 1;

		for (int num : A) {
			if ((mask & num) != 0)
				x ^= num;
			else
				y ^= num;
		}
		res.add(x);
		res.add(y);
		return res;
	}

	public int atoi(String str) {
		// write your code here
		str = str.trim();
		if (str.length() == 0)
			return 0;

		boolean neg = false;
		boolean overflow = false;
		int i = 0;
		if (str.charAt(i) == '+')
			i++;
		else if (str.charAt(i) == '-') {
			neg = true;
			i++;
		}
		int sum = 0;
		while (i < str.length()) {
			int dig = str.charAt(i) - '0';
			if (dig < 0 || dig > 9)
				break;
			if ((Integer.MAX_VALUE - dig) / 10 < sum) {
				overflow = true;
				break;
			}
			sum = sum * 10 + dig;
			i++;
		}
		if (neg) {
			if (overflow)
				return Integer.MIN_VALUE;
			return -sum;
		}
		if (overflow)
			return Integer.MAX_VALUE;
		return sum;
	}

	public String longestCommonPrefix(String[] strs) {
		// write your code here
		if (strs.length == 0)
			return "";
		String s = strs[0];
		for (String str : strs)
			s = str.length() < s.length() ? str : s;

		for (int i = 0; i < s.length(); i++) {
			for (String str : strs) {
				if (s.charAt(i) != str.charAt(i))
					return s.substring(0, i);
			}
		}
		return s;
	}

	public int ladderLength(String start, String end, Set<String> dict) {
		// write your code here
		if (start.equals(end))
			return 0;
		int step = 1;
		Queue<String> que = new LinkedList<String>();
		Set<String> set = new HashSet<String>();
		int curlevel = 0;
		int nextlevel = 0;
		que.add(start);
		curlevel++;
		set.add(start);

		while (!que.isEmpty()) {
			String top = que.remove();
			set.add(top);
			curlevel--;
			// if(dict.contains(top))
			// dict.remove(top);
			if (top.equals(end))
				return step;
			char[] chars = top.toCharArray();
			for (int i = 0; i < chars.length; i++) {
				char ch = chars[i];
				for (char c = 'a'; c <= 'z'; c++) {
					if (c != ch) {
						chars[i] = c;
						String s = new String(chars);
						if (dict.contains(s) && !set.contains(s)) {
							que.add(s);
							set.add(s);
							nextlevel++;
						}
					}
				}
				chars[i] = ch;
			}
			if (curlevel == 0) {
				curlevel = nextlevel;
				nextlevel = 0;
				step++;
			}
		}
		return step;
	}

	/**
	 * @param s
	 *            : A string
	 * @param p
	 *            : A string includes "." and "*"
	 * @return: A boolean
	 */
	public boolean isMatch(String s, String p) {
		// write your code here
		if (p.length() == 0)
			return s.length() == 0;
		if (p.length() == 1) {
			if (s.length() == 1
					&& (p.charAt(0) == s.charAt(0) || p.charAt(0) == '.'))
				return true;
			else
				return false;
		}
		if (p.charAt(1) != '*') {
			if (s.length() > 0
					&& (p.charAt(0) == s.charAt(0) || p.charAt(0) == '.'))
				return isMatch(s.substring(1), p.substring(1));
			return false;
		} else {
			while (s.length() > 0
					&& (p.charAt(0) == s.charAt(0) || p.charAt(0) == '.')) {
				if (isMatch(s, p.substring(2)))
					return true;
				s = s.substring(1);
			}
			return isMatch(s, p.substring(2));
		}
	}

	/**
	 * @param s
	 *            : A string
	 * @param p
	 *            : A string includes "?" and "*"
	 * @return: A boolean
	 */
	public boolean isMatchWild(String s, String p) {
		// write your code here
		if (p.length() == 0)
			return s.length() == 0;
		int i = 0;
		int j = 0;
		int star = -1;
		int sp = 0;

		while (i < s.length()) {
			if (j < p.length()
					&& (s.charAt(i) == p.charAt(j) || p.charAt(j) == '?')) {
				i++;
				j++;
			} else if (j < p.length() && p.charAt(j) == '*') {
				star = j++;
				sp = i;
			} else if (star != -1) {
				j = star + 1;
				i = ++sp;
			} else
				return false;
		}

		while (j < p.length() && p.charAt(j) == '*')
			j++;
		return j == p.length();
	}

	public ArrayList<DirectedGraphNode> topSort(
			ArrayList<DirectedGraphNode> graph) {
		// write your code here
		ArrayList<DirectedGraphNode> res = new ArrayList<DirectedGraphNode>();
		if (graph == null || graph.size() == 0)
			return res;
		HashSet<DirectedGraphNode> visited = new HashSet<DirectedGraphNode>();

		for (DirectedGraphNode node : graph) {
			if (!visited.contains(node))
				topSortUtil(node, visited, res);
		}
		Collections.reverse(res);
		return res;
	}

	public void topSortUtil(DirectedGraphNode node,
			HashSet<DirectedGraphNode> visited, ArrayList<DirectedGraphNode> res) {
		if (visited.contains(node))
			return;
		visited.add(node);
		ArrayList<DirectedGraphNode> neighbors = node.neighbors;
		for (int i = 0; i < neighbors.size(); i++) {
			topSortUtil(neighbors.get(i), visited, res);
		}
		res.add(node);
	}

	public int updateBits(int n, int m, int i, int j) {
		// write your code here
		i = i < 0 ? 0 : i;
		i = i > 31 ? 31 : i;
		j = j < 0 ? 0 : j;
		j = j > 31 ? 31 : j;
		if (i > j) {
			int t = i;
			i = j;
			j = t;
		}
		int mask1 = 0;
		if (j == 31)
			mask1 = ~0;
		else
			mask1 = (1 << (j + 1)) - 1;
		int mask2 = (1 << i) - 1;
		int mask = ~(mask1 - mask2);

		return (n & mask) | (m << i);
	}

	public TreeNode removeNode(TreeNode root, int value) {
		// write your code here
		if (root == null)
			return null;
		if (root.val > value)
			root.left = removeNode(root.left, value);
		else if (root.val < value)
			root.right = removeNode(root.right, value);
		else {
			if (root.left == null)
				return root.right;
			else if (root.right == null)
				return root.left;
			else {
				// TreeNode t = root;
				// root = getLeftMost(t.right);
				// root.right = deleteMin(t.right);
				// root.left = t.left;
				TreeNode t = root;
				root = getRightMost(t.left);
				root.left = deleteMax(t.left);
				root.right = t.right;
			}
		}
		return root;
	}

	// private TreeNode getLeftMost(TreeNode root){
	// while(root.left!=null)
	// root=root.left;
	// return root;
	// }

	// private TreeNode deleteMin(TreeNode root) {
	// if (root.left == null) return root.right;
	// root.left = deleteMin(root.left);
	// return root;
	// }

	private TreeNode getRightMost(TreeNode root) {
		while (root.right != null)
			root = root.right;
		return root;
	}

	private TreeNode deleteMax(TreeNode root) {
		if (root.right == null)
			return root.left;
		root.right = deleteMax(root.right);
		return root;
	}

	public int MinAdjustmentCost(ArrayList<Integer> A, int target) {
		// write your code here
		int n = A.size();
		// dp[i][j] means update index i with min cost j.
		int[][] dp = new int[n][101];

		for (int i = 0; i < n; i++) {
			for (int j = 1; j <= 100; j++) {
				if (i == 0)
					dp[i][j] = Math.abs(j - A.get(i));
				else {
					dp[i][j] = Integer.MAX_VALUE;
					for (int k = 1; k <= 100; k++) {
						if (Math.abs(j - k) <= target) {
							int dif = Math.abs(j - A.get(i)) + dp[i - 1][k];
							dp[i][j] = Math.min(dp[i][j], dif);
						}
					}
				}
			}
		}
		int min = Integer.MAX_VALUE;
		for (int i = 1; i <= 100; i++)
			min = Math.min(min, dp[n - 1][i]);
		return min;
	}

	public List<List<String>> findLadders(String start, String end,
			Set<String> dict) {
		// write your code here
		List<List<String>> res = new ArrayList<List<String>>();
		if (dict.size() == 0)
			return res;
		Queue<String> que = new LinkedList<String>();
		que.add(start);
		List<List<String>> paths = new ArrayList<List<String>>();
		List<String> path = new ArrayList<String>();
		path.add(start);
		paths.add(path);

		HashMap<String, List<List<String>>> map = new HashMap<String, List<List<String>>>();
		map.put(start, paths);

		while (!que.isEmpty()) {
			String str = que.remove();
			List<List<String>> allPaths = map.get(str);
			if (str.equals(end)) {
				return allPaths;
			}
			char[] chars = str.toCharArray();
			for (int i = 0; i < chars.length; i++) {
				char t = chars[i];
				for (char c = 'a'; c <= 'z'; c++) {
					if (t != c) {
						chars[i] = c;
						String newStr = new String(chars);
						if (dict.contains(newStr)) {
							List<List<String>> toAdd = new ArrayList<List<String>>();
							if (!map.containsKey(newStr)) {
								que.add(newStr);
								for (List<String> p : allPaths) {
									List<String> newPath = new ArrayList<String>(
											p);
									newPath.add(newStr);
									toAdd.add(newPath);
								}
								map.put(newStr, toAdd);
							} else if (map.get(newStr).get(0).size() == map
									.get(str).get(0).size() + 1) {
								for (List<String> p : allPaths) {
									List<String> newPath = new ArrayList<String>(
											p);
									newPath.add(newStr);
									toAdd.add(newPath);
								}
								toAdd.addAll(map.get(newStr));
								map.put(newStr, toAdd);
							}
						}
					}
				}
				chars[i] = t;
			}
		}
		return res;
	}

	public SegmentTreeNode build(int start, int end) {
		// write your code here
		if (end < start)
			return null;
		if (start == end)
			return new SegmentTreeNode(start, end);
		SegmentTreeNode root = new SegmentTreeNode(start, end);
		root.left = build(start, (start + end) / 2);
		root.right = build((start + end) / 2 + 1, end);
		return root;
	}

	public int query(SegmentTreeNode root, int start, int end) {
		// write your code here
		if (start <= root.start && end >= root.end)
			return root.max;
		if (root.start > end || root.end < start)
			return Integer.MIN_VALUE;
		return Math.max(query(root.left, start, end),
				query(root.right, start, end));
	}

	public void modify(SegmentTreeNode root, int index, int value) {
		// write your code here
		if (root == null)
			return;
		if (index < root.start || index > root.end)
			return;

		if (index <= (root.start + root.end) / 2)
			modify(root.left, index, value);
		else
			modify(root.right, index, value);
		if (root.start == index && root.end == index)
			root.max = value;
		else
			root.max = Math.max(root.left.max, root.right.max);
	}

	public ArrayList<Integer> intervalMinNumber(int[] A,
			ArrayList<Interval> queries) {
		// write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();
		SegmentTreeNode root = build(A, 0, A.length - 1);
		for (int i = 0; i < queries.size(); i++) {
			Interval interval = queries.get(i);
			int queryRs = query(root, interval.start, interval.end);
			res.add(queryRs);
		}
		return res;
	}

	public SegmentTreeNode build(int[] A, int start, int end) {
		// write your code here
		if (end < start)
			return null;
		if (start == end)
			return new SegmentTreeNode(start, end, A[start]);
		int max = A[start];
		for (int i = start + 1; i <= end; i++)
			max = Math.max(A[i], max);
		SegmentTreeNode root = new SegmentTreeNode(start, end, max);
		root.left = build(A, start, (start + end) / 2);
		root.right = build(A, (start + end) / 2 + 1, end);
		return root;
	}

	public void inorder(SegmentTreeNode root) {
		if (root == null)
			return;
		inorder(root.left);
		System.out.print("[" + root.start + ", " + root.end + ", " + root.max
				+ "]" + " ");
		inorder(root.right);
	}

	public ArrayList<Long> intervalSum(int[] A, ArrayList<Interval> queries) {
		// write your code here
		ArrayList<Long> res = new ArrayList<Long>();
		if (A.length == 0)
			return res;
		SegmentTreeNode root = build2(A, 0, A.length - 1);
		for (int i = 0; i < queries.size(); i++) {
			Interval interval = queries.get(i);
			long queryRes = query2(root, interval.start, interval.end);
			res.add(queryRes);
		}
		return res;
	}

	public SegmentTreeNode build2(int[] A, int start, int end) {
		// write your code here
		if (end < start)
			return null;
		if (start == end)
			return new SegmentTreeNode(start, end, A[start]);
		int sum = 0;
		for (int i = start; i <= end; i++)
			sum += A[i];

		SegmentTreeNode root = new SegmentTreeNode(start, end, sum);
		root.left = build2(A, start, (start + end) / 2);
		root.right = build2(A, (start + end) / 2 + 1, end);
		return root;
	}

	public int query2(SegmentTreeNode root, int start, int end) {
		// write your code here
		if (start <= root.start && end >= root.end)
			return root.max;
		if (root.start > end || root.end < start)
			return 0;
		return query2(root.left, start, end) + query2(root.right, start, end);
	}

	public int maxProduct(int[] nums) {
		// write your code here
		if (nums.length == 0)
			return 0;
		int max = nums[0];
		int curmax = nums[0];
		int curmin = nums[0];

		for (int i = 1; i < nums.length; i++) {
			int a = curmax * nums[i];
			int b = curmin * nums[i];
			curmax = Math.max(Math.max(a, b), nums[i]);
			curmin = Math.min(Math.min(a, b), nums[i]);
			max = Math.max(curmax, max);
		}
		return max;
	}

	public boolean hasRoute(ArrayList<DirectedGraphNode> graph,
			DirectedGraphNode s, DirectedGraphNode t) {
		// write your code here
		if (s == t)
			return true;
		Queue<DirectedGraphNode> que = new LinkedList<DirectedGraphNode>();
		Set<DirectedGraphNode> visited = new HashSet<DirectedGraphNode>();
		que.add(s);
		visited.add(s);
		while (!que.isEmpty()) {
			DirectedGraphNode node = que.remove();
			ArrayList<DirectedGraphNode> neighbors = node.neighbors;
			for (DirectedGraphNode n : neighbors) {
				if (n == t)
					return true;
				if (!visited.contains(n)) {
					que.add(n);
					visited.add(n);
				}
			}
		}
		return false;
	}

	public ArrayList<Integer> maxSlidingWindow(int[] nums, int k) {
		// write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();
		if (nums.length == 0)
			return res;
		PriorityQueue<Pair> que = new PriorityQueue<Pair>(k,
				new Comparator<Pair>() {

					@Override
					public int compare(Pair o1, Pair o2) {
						// TODO Auto-generated method stub
						return o2.val - o1.val;
					}
				});
		for (int i = 0; i < k; i++) {
			que.add(new Pair(nums[i], i));
		}
		for (int i = k; i < nums.length; i++) {
			Pair p = que.peek();
			res.add(p.val);
			while (p != null && p.index <= i - k) {
				que.poll();
				p = que.peek();
			}
			que.add(new Pair(nums[i], i));
		}
		res.add(que.peek().val);
		return res;
	}

	public ArrayList<Integer> maxSlidingWindow2(int[] nums, int k) {
		// write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();
		if (nums.length == 0)
			return res;
		Deque<Integer> que = new ArrayDeque<Integer>();
		for (int i = 0; i < k; i++) {
			while (!que.isEmpty() && nums[i] >= nums[que.getLast()])
				que.pollLast();
			que.offerLast(i);
		}

		for (int i = k; i < nums.length; i++) {
			res.add(nums[que.getFirst()]);
			while (!que.isEmpty() && nums[i] >= nums[que.getLast()])
				que.pollLast();
			while (!que.isEmpty() && que.getFirst() <= i - k)
				que.pollFirst();
			que.offerLast(i);
		}
		res.add(nums[que.pollFirst()]);
		return res;
	}

	// Answer: just simulate the move
	// 1)start at i=1, j=1
	// 2)move down once(i++) OR move right if you are at the bottom side
	// 3)move in north east direction until you are reached top or right side
	// 4)move right once if you are at top side OR move down once if you are at
	// right side
	// 5)move in south west direction until you are reached bottom or left side
	// 5)go to step2 if you are still in the range

	public int[] printZMatrix(int[][] matrix) {
		// write your code here
		int m = matrix.length;
		if (m == 0)
			return null;
		int n = matrix[0].length;
		int[] res = new int[m * n];
		int k = 0;
		int i = 0, j = 0;
		do {
			res[k++] = matrix[i][j];
			if (j < n - 1)
				j++;
			else if (i < m - 1)
				i++;
			else
				break;
			// sw direction
			while (i < m - 1 && j > 0) {
				res[k++] = matrix[i][j];
				i++;
				j--;
			}
			res[k++] = matrix[i][j];
			if (j == 0 && i < m - 1)
				i++;
			else
				j++;

			while (i > 0 && j < n - 1) {
				res[k++] = matrix[i][j];
				i--;
				j++;
			}

		} while (i < m && j < n);
		return res;

	}

	void printZMatrix2(int[][] matrix) {
		int row = matrix.length;
		int col = matrix[0].length;
		int i = 0, j = 0;
		do {
			System.out.print(matrix[i][j] + " ");
			if (i < row - 1)
				i++;
			else if (j < col - 1)
				j++;
			else
				// already finished printing
				break;
			// NE(north east) direction
			while (i > 0 && j < col - 1) {
				System.out.print(matrix[i][j] + " ");
				i--;
				j++;
			}
			System.out.print(matrix[i][j] + " ");
			if (i == 0 && j < col - 1)
				j++;
			else
				i++;
			while (i < row - 1 && j > 0) {// SW direction
				System.out.print(matrix[i][j] + " ");
				i++;
				j--;
			}
		} while (i <= row - 1 && j <= col - 1);
		System.out.println();
	}

	public ListNode insertionSortList(ListNode head) {
		// write your code here
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;

		ListNode cur = head.next;
		ListNode last = head;
		while (cur != null) {
			ListNode node = dummy.next;
			ListNode pre = dummy;
			while (node != cur && node.val < cur.val) {
				pre = node;
				node = node.next;
			}
			if (node != cur) {
				last.next = cur.next;
				pre.next = cur;
				cur.next = node;
			} else
				last = cur;
			cur = last.next;
		}
		return dummy.next;
	}

	public int evaluateExpression(String[] expression) {
		// write your code here
		// if(expression.length==0)
		// return 0;
		Stack<Integer> values = new Stack<Integer>();
		Stack<String> ops = new Stack<String>();
		for (int i = 0; i < expression.length; i++) {
			String token = expression[i];
			if (Character.isDigit(token.charAt(0)))
				values.push(Integer.parseInt(token));
			else if (token.equals("("))
				ops.push(token);
			else if (token.equals(")")) {
				while (!ops.isEmpty() && !ops.peek().equals("(")) {
					values.push(applyOp(ops.pop(), values.pop(), values.pop()));
				}
				ops.pop();
			} else if (token.equals("+") || token.equals("-")
					|| token.equals("*") || token.equals("/")) {
				while (!ops.isEmpty() && hasPrecedence(token, ops.peek()))
					values.push(applyOp(ops.pop(), values.pop(), values.pop()));
				ops.push(token);
			}
		}
		while (!ops.isEmpty()) {
			values.push(applyOp(ops.pop(), values.pop(), values.pop()));
		}
		if (!values.isEmpty())
			return values.pop();
		return 0;
	}

	public int applyOp(String op, int val1, int val2) {
		if (op.equals("+"))
			return val1 + val2;
		else if (op.equals("-"))
			return val2 - val1;
		else if (op.equals("*"))
			return val1 * val2;
		return val2 / val1;
	}

	public boolean hasPrecedence(String op1, String op2) {
		if (op2.equals("(") || op2.equals(""))
			return false;
		if ((op1.equals("*") || op1.equals("/"))
				&& (op2.equals("+") || op2.equals("-")))
			return false;
		return true;
	}

	public ArrayList<Integer> medianSlidingWindow(int[] nums, int k) {
		// write your code here
		PriorityQueue<Integer> minHeap = new PriorityQueue<Integer>();
		PriorityQueue<Integer> maxHeap = new PriorityQueue<Integer>(k,
				new Comparator<Integer>() {
					@Override
					public int compare(Integer i1, Integer i2) {
						return i2 - i1;
					}
				});
		ArrayList<Integer> res = new ArrayList<Integer>();
		for (int i = 0; i < nums.length; i++) {
			if (i >= k) {
				if (minHeap.contains(nums[i - k]))
					minHeap.remove(nums[i - k]);
				else
					maxHeap.remove(nums[i - k]);
			}
			// Balance smaller half and larger half.
			if (maxHeap.isEmpty() || nums[i] > maxHeap.peek()) {
				minHeap.offer(nums[i]);
				if (minHeap.size() > maxHeap.size() + 1) {
					maxHeap.offer(minHeap.poll());
				}
			} else {
				maxHeap.offer(nums[i]);
				if (maxHeap.size() > minHeap.size()) {
					minHeap.offer(maxHeap.poll());
				}
			}

			// If window is full, get the median from 2 BST.
			if (i >= k - 1) {
				res.add(minHeap.size() == maxHeap.size() ? maxHeap.peek()
						: minHeap.peek());
			}
		}

		return res;
	}

	public ArrayList<Integer> countOfSmallerNumber(int[] A, int[] queries) {
		ArrayList<Integer> res = new ArrayList<Integer>();
		if (A.length == 0 || queries.length == 0)
			return res;
		Arrays.sort(A);
		for (int i = 0; i < queries.length; i++) {
			int count = binaryCount(A, queries[i]);
			res.add(count);
		}
		return res;
	}

	public int binaryCount(int[] A, int target) {
		int beg = 0;
		int end = A.length - 1;

		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A[mid] == target) {
				while (mid > 0 && A[mid - 1] == target)
					mid--;
				return mid;
			}
			if (A[mid] > target)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		return beg;
	}

	public int smallestDifference(int[] A, int[] B) {
		// write your code here
		Arrays.sort(A);
		Arrays.sort(B);
		int min = Integer.MAX_VALUE;

		int i = 0;
		int j = 0;
		while (i < A.length && j < B.length) {
			min = Math.min(min, Math.abs(A[i] - B[j]));
			if (A[i] == B[j])
				return 0;
			else if (A[i] > B[j]) {
				j++;
			} else {
				i++;
			}
		}
		return min;
	}

	public int lengthOfLongestSubstring(String s) {
		// write your code here
		if (s.length() < 2)
			return s.length();
		int max = 0;
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		int start = 0;

		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (!map.containsKey(c)) {
				map.put(c, i);
				max = Math.max(max, i - start + 1);
			} else {
				int dup = map.get(c);
				for (int j = dup; j >= start; j--) {
					map.remove(s.charAt(j));
				}
				map.put(c, i);
				start = dup + 1;
			}
		}
		return max;
	}

	public int maxArea(int[] heights) {
		// write your code here
		int max = 0;
		int i = 0;
		int j = heights.length - 1;
		while (i < j) {
			int area = Math.min(heights[i], heights[j]) * (j - i);
			max = Math.max(max, area);
			if (heights[i] < heights[j])
				i++;
			else
				j--;
		}
		return max;
	}

	public int lengthOfLongestSubstringKDistinct(String s, int k) {
		// write your code here
		if (s.length() <= k)
			return s.length();
		int max = 0;
		int start = 0;
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();

		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (map.containsKey(c))
				map.put(c, map.get(c) + 1);
			else {
				if (map.size() < k)
					map.put(c, 1);
				else {
					while (map.size() == k && start < i) {
						char t = s.charAt(start);
						int count = map.get(t);
						if (count > 1)
							map.put(t, count - 1);
						else
							map.remove(t);
						start++;
					}
					map.put(c, 1);
				}
			}
			max = Math.max(max, i - start + 1);
		}
		return max;
	}

	public List<Interval> merge(List<Interval> intervals) {
		// write your code here
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
			Interval i1 = res.get(res.size() - 1);
			Interval i2 = intervals.get(i);
			if (i1.end < i2.start)
				res.add(i2);
			else
				i1.end = Math.max(i1.end, i2.end);
		}
		return res;
	}

	public void deleteMiddleNode(ListNode node) {
		// write your code here
		if (node == null || node.next == null) {
			node = null;
			return;
		}
		ListNode fast = node;
		ListNode slow = node;
		ListNode pre = node;

		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			// if(fast==null)
			// break;
			pre = slow;
			slow = slow.next;
		}

		pre.next = slow.next;
		System.out.println(slow.val);
		System.out.println(pre.val);
	}

	public int minSubArrayLen(int s, int[] nums) {
		int min = nums.length + 1;
		int sum = 0;
		int start = 0;
		for (int i = 0; i < nums.length; i++) {
			sum += nums[i];
			while (sum >= s) {
				min = Math.min(min, i - start + 1);
				sum -= nums[start++];
			}
		}
		if (min > nums.length)
			return 0;
		return min;
	}

	public String intToRoman(int n) {
		// Write your code here
		if (n < 1 || n > 3999)
			return "";
		String[] roman = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X",
				"IX", "V", "IV", "I" };
		int[] nums = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };

		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < nums.length; i++) {
			while (n >= nums[i]) {
				sb.append(roman[i]);
				n -= nums[i];
			}
		}
		return sb.toString();
	}

	public int romanToInt(String s) {
		// Write your code here
		int res = 0;
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		map.put('I', 1);
		map.put('V', 5);
		map.put('X', 10);
		map.put('L', 50);
		map.put('C', 100);
		map.put('D', 500);
		map.put('M', 1000);

		for (int i = 0; i < s.length(); i++) {
			res += sign(s, i, map) * map.get(s.charAt(i));
		}
		return res;
	}

	public int sign(String s, int i, HashMap<Character, Integer> map) {
		if (i == s.length() - 1)
			return 1;
		if (map.get(s.charAt(i)) < map.get(s.charAt(i + 1)))
			return -1;
		return 1;
	}

	public int fibonacci(int n) {
		// write your code here
		if (n == 1)
			return 0;
		if (n == 2)
			return 1;
		int first = 0;
		int second = 1;
		int total = 0;
		for (int i = 3; i <= n; i++) {
			total = first + second;
			first = second;
			second = total;
		}
		return total;
	}

	public ArrayList<Integer> continuousSubarraySum(int[] A) {
		// Write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();
		if (A.length == 0)
			return res;
		res.add(0);
		res.add(0);
		int sum = 0;
		int max = 0;
		int start = 0;
		int end = 0;
		for (int i = 0; i < A.length; i++) {
			sum += A[i];
			if (sum < 0) {
				sum = 0;
				start = i + 1;
				end = i + 1;
			}
			if (sum > max) {
				max = sum;
				end = i;
				res.set(0, start);
				res.set(1, end);
			}
		}
		if (max == 0) {
			max = Integer.MIN_VALUE;
			for (int i = 0; i < A.length; i++) {
				if (A[i] > max) {
					max = A[i];
					start = i;
					end = i;
				}
			}
			res.set(0, start);
			res.set(1, end);
		}
		return res;
	}

	public ArrayList<Integer> continuousSubarraySum2(int[] A) {
		ArrayList<Integer> res = new ArrayList<Integer>();
		if (A == null || A.length == 0) {
			return res;
		}
		int sum = A[0];
		int max = sum;
		int start = 0, end = 0;
		res.add(0);
		res.add(0);
		for (int i = 1; i < A.length; i++) {
			if (sum > max) {
				res.set(0, start);
				res.set(1, i - 1);
				max = sum;
			}
			if (sum < 0) {
				sum = 0;
				start = i;
				end = i;
			}
			sum += A[i];
		}
		if (sum > max) {
			res.set(0, start);
			res.set(1, A.length - 1);
		}
		return res;
	}

	ArrayList<String> longestWords(String[] dictionary) {
		// write your code here
		ArrayList<String> res = new ArrayList<String>();
		int max = 0;
		for (String s : dictionary) {
			if (s.length() > max) {
				res.clear();
				res.add(s);
				max = s.length();
			} else if (s.length() == max)
				res.add(s);
		}
		return res;
	}

	public int countOfAirplanes(List<Interval> airplanes) {
		// write your code here
		int n = airplanes.size();
		if (n < 2)
			return n;

		int max = 1;
		Collections.sort(airplanes, new IntervalComp());
		PriorityQueue<Interval> heap = new PriorityQueue<Interval>(n,
				new HeapComp());

		for (int i = 0; i < n; i++) {
			Interval cur = airplanes.get(i);
			if (heap.isEmpty() || cur.start < heap.peek().end)
				heap.offer(cur);
			else {
				while (!heap.isEmpty() && cur.start >= heap.peek().end)
					heap.poll();
				heap.offer(cur);
			}
			max = Math.max(max, heap.size());
		}
		return max;
	}

	class IntervalComp implements Comparator<Interval> {
		@Override
		public int compare(Interval i1, Interval i2) {
			// if(i1.start==i2.start)
			// return i1.end-i2.end;
			return i1.start - i2.start;
		}
	}

	class HeapComp implements Comparator<Interval> {
		@Override
		public int compare(Interval i1, Interval i2) {
			return i1.end - i2.end;
		}
	}

	// public int replaceBlank(char[] string, int length) {
	// int spaces = 0;
	// for(int i=0; i<length; i++){
	// if(string[i]==' ') spaces++;
	// }
	// if(spaces==0) return length;
	// int new_length = length + 2* spaces;
	// int j= new_length-1;
	// for(int i=length-1; i>=0; i--){
	// if(string[i] != ' '){
	// string[j] = string[i];
	// j--;
	// }else{
	// string[j] = '0';
	// string[j-1] = '2';
	// string[j-2] = '%';
	// j=j-3;
	// }
	// }
	// return new_length;
	// }

	public boolean isValidParentheses(String s) {
		// Write your code here
		if (s.length() % 2 != 0)
			return false;
		Stack<Character> stk = new Stack<Character>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (c == '(' || c == '[' || c == '{')
				stk.push(c);
			else if (!stk.isEmpty()
					&& (c == '(' && stk.peek() == ')' || c == '['
							&& stk.peek() == ']' || c == '{'
							&& stk.peek() == '}'))
				stk.pop();
			else
				return false;
		}
		return stk.isEmpty();
	}

	public String simplifyPath(String path) {
		// Write your code here
		String[] strs = path.split("/");
		Stack<String> stk = new Stack<String>();
		String sb = "";

		for (int i = 0; i < strs.length; i++) {
			String s = strs[i];
			if (s.equals("") || s.equals("."))
				continue;
			else if (s.equals("..")) {
				if (!stk.isEmpty())
					stk.pop();
			} else
				stk.push(s);
		}
		if (stk.isEmpty())
			return "/";
		while (!stk.isEmpty()) {
			sb = "/" + stk.pop() + sb;
		}
		return sb;
	}

	public int countOnes(int num) {
		// write your code here
		int count = 0;
		while (num > 0) {
			if ((num & 1) == 1)
				count++;
			num >>= 1;
		}
		return count;
	}

	public int evalRPN(String[] tokens) {
		// Write your code here
		Stack<Integer> stk = new Stack<Integer>();
		for (int i = 0; i < tokens.length; i++) {
			String token = tokens[i];
			if (token.equals("+") || token.equals("-") || token.equals("*")
					|| token.equals("/")) {
				int op1 = stk.pop();
				int op2 = stk.pop();

				if (token.equals("+"))
					stk.push(op1 + op2);
				else if (token.equals("-"))
					stk.push(op2 - op1);
				else if (token.equals("*"))
					stk.push(op1 * op2);
				else
					stk.push(op2 / op1);
			} else
				stk.push(Integer.parseInt(token));
		}
		return stk.pop();
	}

	public String countAndSay(int n) {
		// Write your code here
		String s = "1";
		for (int i = 1; i < n; i++) {
			char c = s.charAt(0);
			int count = 1;
			String t = "";
			for (int j = 1; j < s.length(); j++) {
				if (c == s.charAt(j))
					count++;
				else {
					t = t + count + c;
					c = s.charAt(j);
					count = 1;
				}
			}
			s = t + count + c;
		}
		return s;
	}

	public String addBinary(String a, String b) {
		// Write your code here
		if (a.length() == 0 || b.length() == 0)
			return a.length() == 0 ? b : a;
		int i = a.length() - 1;
		int j = b.length() - 1;

		String res = "";
		int carry = 0;
		while (i >= 0 || j >= 0) {
			int num1 = i >= 0 ? a.charAt(i) - '0' : 0;
			int num2 = j >= 0 ? b.charAt(j) - '0' : 0;
			int sum = num1 + num2 + carry;
			carry = sum / 2;
			sum = sum % 2;
			res = sum + res;
			i--;
			j--;
		}
		return carry == 1 ? "1" + res : res;
	}

	public int candy(int[] ratings) {
		// Write your code here
		int[] candy = new int[ratings.length];
		Arrays.fill(candy, 1);
		for (int i = 1; i < ratings.length; i++) {
			if (ratings[i] > ratings[i - 1])
				candy[i] = candy[i - 1] + 1;
		}

		for (int i = ratings.length - 2; i >= 0; i--) {
			if (ratings[i] > ratings[i + 1])
				candy[i] = Math.max(candy[i], candy[i + 1] + 1);
		}
		int count = 0;

		for (int i = 0; i < ratings.length; i++) {
			count += candy[i];
		}
		return count;
	}

	public int[] plusOne(int[] digits) {
		// Write your code here
		int carry = 1;
		for (int i = digits.length - 1; i >= 0; i--) {
			int sum = digits[i] + carry;
			carry = sum / 10;
			sum = sum % 10;
			digits[i] = sum;
		}
		if (carry == 1) {
			int[] res = new int[digits.length + 1];
			res[0] = 1;
			for (int j = 1; j < res.length; j++) {
				res[j] = digits[j - 1];
			}
			return res;
		}
		return digits;
	}

	public int minSubArray2(ArrayList<Integer> nums) {
		if (nums.size() == 0)
			return 0;
		int res = nums.get(0);
		int curMin = nums.get(0);

		for (int i = 1; i < nums.size(); i++) {
			curMin = Math.min(nums.get(i), curMin + nums.get(i));
			res = Math.min(res, curMin);
		}
		return res;
	}

	public boolean wordPattern(String pattern, String str) {
		String[] strs = str.split(" ");
		if (pattern.length() != strs.length)
			return false;
		Map<String, Character> map1 = new HashMap<String, Character>();
		Map<Character, String> map2 = new HashMap<Character, String>();
		for (int i = 0; i < strs.length; i++) {
			if (map1.containsKey(strs[i])) {
				if (map1.get(strs[i]) != pattern.charAt(i))
					return false;
			} else {
				map1.put(strs[i], pattern.charAt(i));
			}

			if (map2.containsKey(pattern.charAt(i))) {
				if (!map2.get(pattern.charAt(i)).equals(strs[i]))
					return false;
			} else {
				map2.put(pattern.charAt(i), strs[i]);
			}
		}
		return true;
	}

	public int addDigits(int num) {
		while (num >= 10) {
			int sum = 0;
			int t = num;
			while (t > 0) {
				sum += t % 10;
				t /= 10;
			}
			num = sum;
		}
		return num;
	}

	public String longestPalindrome(String s) {
		if (s.length() < 2)
			return s;
		int longest = 1;
		String res = "";
		for (int i = 1; i < s.length(); i++) {
			int left = i - 1;
			int right = i + 1;
			while (left >= 0 && right < s.length()
					&& s.charAt(left) == s.charAt(right)) {
				left--;
				right++;
			}
			if (right - left - 1 > longest) {
				longest = right - left - 2;
				res = s.substring(left + 1, right);
			}

			left = i - 1;
			right = i;
			while (left >= 0 && right < s.length()
					&& s.charAt(left) == s.charAt(right)) {
				left--;
				right++;
			}
			System.out.println(left + " " + right + ", " + (right - left - 1));
			if (right - left - 1 > longest) {
				longest = right - left - 2;
				res = s.substring(left + 1, right);
			}
		}
		System.out.println(longest);
		return res;
	}

	public ArrayList<String> restoreIpAddresses(String s) {
		// Write your code here
		ArrayList<String> res = new ArrayList<String>();
		if (s.length() < 4 || s.length() > 12)
			return res;
		dfs(s, "", res, 0);
		return res;
	}

	public void dfs(String s, String sol, ArrayList<String> res, int dep) {
		if (dep == 3 && isValidNum(s)) {
			res.add(sol + s);
			return;
		}
		for (int i = 1; i < 4 && i < s.length(); i++) {
			if (isValidNum(s.substring(0, i))) {
				dfs(s.substring(i), sol + s.substring(0, i) + ".", res, dep + 1);
			}
		}
	}

	public boolean isValidNum(String s) {
		if (s.charAt(0) == '0')
			return s.equals("0");
		int num = Integer.parseInt(s);
		return num >= 1 && num <= 255;
	}

	public ArrayList<String> generateParenthesis(int n) {
		// Write your code here
		ArrayList<String> res = new ArrayList<String>();
		generateParenthesis(n, 0, 0, "", res);
		return res;
	}

	public void generateParenthesis(int n, int left, int right, String sol,
			ArrayList<String> res) {
		if (left == n && left == right) {
			res.add(sol);
		}
		if (left < n)
			generateParenthesis(n, left + 1, right, sol + "(", res);
		if (right < left)
			generateParenthesis(n, left, right + 1, sol + ")", res);
	}

	public void surroundedRegions(char[][] board) {
		// Write your code here
		int m = board.length;
		if (m == 0)
			return;
		int n = board[0].length;
		Queue<Integer> que = new LinkedList<Integer>();
		for (int i = 0; i < m; i++) {
			if (board[i][0] == 'O') {
				que.offer(i * n);
				bfsRegions(board, que);
			}
			if (board[i][n - 1] == 'O') {
				que.offer(i * n + n - 1);
				bfsRegions(board, que);
			}
		}

		for (int i = 0; i < n; i++) {
			if (board[0][i] == 'O') {
				que.offer(i);
				bfsRegions(board, que);
			}
			if (board[m - 1][i] == 'O') {
				que.offer((m - 1) * n + i);
				bfsRegions(board, que);
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

	// public void dfsRegions(char[][] board, int i, int j){
	// if(i<0||i>=board.length||j<0||j>=board[0].length||board[i][j]=='X'||board[i][j]=='#')
	// return;
	// board[i][j]='#';
	// dfsRegions(board, i+1, j);
	// dfsRegions(board, i-1, j);
	// dfsRegions(board, i, j+1);
	// dfsRegions(board, i, j-1);

	// }

	public boolean isHappy(int n) {
		// Write your code here
		Set<Integer> set = new HashSet<Integer>();
		set.add(n);

		while (n > 1) {
			int sum = 0;
			while (n > 0) {
				sum += Math.pow(n % 10, 2);
				n /= 10;
			}
			if (!set.add(sum))
				return false;
			n = sum;
		}
		return true;
	}

	public void bfsRegions(char[][] board, Queue<Integer> que) {
		int n = board[0].length;
		while (!que.isEmpty()) {
			int first = que.poll();
			int row = first / n;
			int col = first % n;
			board[row][col] = '#';
			if (row + 1 < board.length && board[row + 1][col] == 'O')
				que.offer((row + 1) * n + col);
			if (row - 1 >= 0 && board[row - 1][col] == 'O')
				que.offer((row - 1) * n + col);
			if (col + 1 < n && board[row][col + 1] == 'O')
				que.offer(row * n + col + 1);
			if (col - 1 >= 0 && board[row][col - 1] == 'O')
				que.offer(row * n + col - 1);
		}
	}

	public ListNode removeElements(ListNode head, int val) {
		// Write your code here
		// ListNode dummy=new ListNode(0);
		// dummy.next=head;
		// ListNode pre=dummy, cur=head;
		// while(cur!=null){
		// while(cur!=null&&cur.val==val){
		// cur=cur.next;
		// }
		// pre.next=cur;
		// pre=cur;
		// if(cur!=null)
		// cur=cur.next;
		// }
		// return dummy.next;
		if (head == null)
			return null;
		head.next = removeElements(head.next, val);
		return head.val == val ? head.next : head;
	}

	public boolean isPalindrome(ListNode head) {
		// Write your code here
		if (head == null || head.next == null)
			return true;
		ListNode fast = head, slow = head;
		while (fast.next != null) {
			fast = fast.next.next;
			if (fast == null)
				break;
			slow = slow.next;
		}
		ListNode secondHead = slow.next;
		slow.next = null;
		secondHead = reverseList(secondHead);

		while (head != null && secondHead != null) {
			if (head.val != secondHead.val)
				return false;
			head = head.next;
			secondHead = secondHead.next;
		}
		return true;
	}

	public ListNode reverseList2(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode pre = null, cur = head;
		while (cur != null) {
			ListNode next = cur.next;
			cur.next = pre;
			pre = cur;
			cur = next;
		}
		return pre;
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

	public ListNode swapPairs(ListNode head) {
		// Write your code here
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode ppre = dummy, pre = head, cur = head.next;
		while (cur != null) {
			ListNode next = cur.next;
			ppre.next = cur;
			cur.next = pre;
			pre.next = next;
			cur = next;
			ppre = pre;
			pre = cur;
			if (cur != null)
				cur = cur.next;
		}
		return dummy.next;
	}

	public double myPow(double x, int n) {
		// Write your code here
		if (n == 0)
			return 1;
		boolean neg = false;
		if (n < 0) {
			neg = true;
			n = -n;
		}

		double res = myPow(x, n / 2);
		if (n % 2 == 0)
			res *= res;
		else
			res *= res * x;
		return neg ? 1 / res : res;
	}

	public int[][] submatrixSum(int[][] matrix) {
		// Write your code here
		int m = matrix.length, n = matrix[0].length;
		int[][] sum = new int[m + 1][n + 1];

		for (int i = 1; i <= m; i++) {
			for (int j = 1; j <= n; j++) {
				sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1]
						+ matrix[i - 1][j - 1];
			}
		}

		int[][] res = new int[2][2];
		for (int i = 0; i < m; i++) {
			for (int j = i + 1; j <= m; j++) {
				Map<Integer, Integer> map = new HashMap<Integer, Integer>();
				for (int k = 0; k <= n; k++) {
					int diff = sum[j][k] - sum[i][k];
					if (map.containsKey(diff)) {
						int col = map.get(diff);
						res[0][0] = i;
						res[0][1] = col;
						res[1][0] = j - 1;
						res[1][1] = k - 1;
						return res;
					} else {
						map.put(diff, k);
					}
				}
			}
		}
		return res;
	}

	public void nextPermutation(int[] nums) {
		// write your code here
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

		int idx = index + 1;
		for (int i = index + 1; i < nums.length; i++) {
			if (nums[i] > nums[index]) {
				idx = i;
			}
		}
		swap2(nums, idx, index);

		int beg = index + 1;
		int end = nums.length - 1;
		while (beg < end) {
			swap2(nums, beg++, end--);
		}
	}

	public void swap2(int[] nums, int i, int j) {
		int t = nums[i];
		nums[i] = nums[j];
		nums[j] = t;
	}

	public long houseRobber(int[] A) {
		// write your code here
		int n = A.length;
		if (n == 0)
			return 0;
		long[] dp = new long[n];
		dp[0] = A[0];
		if (n > 1)
			dp[1] = Math.max(A[0], A[1]);
		for (int i = 2; i < n; i++) {
			dp[i] = Math.max(dp[i - 2] + A[i], dp[i - 1]);
		}
		return dp[n - 1];
	}

	/*
	 * DP[i]表示从i到end能取到的最大value function
	 * 
	 * 当我们走到i时，有两种选择 取values[i] 取values[i] + values[i+1] 1. 我们取了values[i],对手的选择有
	 * values[i+1]或者values[i+1] + values[i+2] 剩下的最大总value分别为DP[i+2]或DP[i+3],
	 * 对手也是理性的所以要让我们得到最小value 所以 value1 = values[i] + min(DP[i+2], DP[i+3]) 2.
	 * 我们取了values[i]和values[i+1] 同理 value2 = values[i] + values[i+1] +
	 * min(DP[i+3], DP[i+4]) 最后
	 * 
	 * DP[I] = max(value1, value2)
	 */

	public boolean firstWillWin(int[] values) {
		// write your code here
		int n = values.length;
		if (n <= 2)
			return true;
		int[] dp = new int[n + 1];
		dp[n] = 0;
		dp[n - 1] = values[n - 1];
		dp[n - 2] = values[n - 1] + values[n - 2];
		dp[n - 3] = values[n - 2] + values[n - 3];

		for (int i = n - 4; i >= 0; i--) {
			dp[i] = values[i] + Math.min(dp[i + 2], dp[i + 3]);
			dp[i] = Math.max(dp[i],
					values[i] + values[i + 1] + Math.min(dp[i + 3], dp[i + 4]));
		}

		int total = 0;
		for (int v : values) {
			total += v;
		}

		return dp[0] > total - dp[0];
	}

	public int reverseInteger(int n) {
		// Write your code here
		boolean neg = false;
		if (n < 0) {
			neg = true;
			n = -n;
		}
		int res = 0;
		while (n > 0) {
			int digit = n % 10;
			if ((Integer.MAX_VALUE - digit) / 10 > res)
				res = res * 10 + digit;
			else
				return 0;
			n /= 10;
		}
		return neg ? -res : res;
	}

	public ArrayList<Integer> grayCode(int n) {
		// Write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();
		if (n == 0) {
			res.add(0);
			return res;
		}
		ArrayList<Integer> t = grayCode(n - 1);

		res.addAll(t);
		for (int i = t.size() - 1; i >= 0; i--) {
			res.add(t.get(i) + (1 << (n - 1)));
		}
		return res;
	}
	
	public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
		// Write your code here
		if (headA == null || headB == null)
			return null;
		int len1 = getLength(headA);
		int len2 = getLength(headB);
		if (len1 < len2) {
			ListNode node = headA;
			headA = headB;
			headB = node;
		}
		for (int i = 0; i < Math.abs(len1 - len2); i++) {
			headA = headA.next;
		}
		while (headA != null && headB != null && headA != headB) {
			headA = headA.next;
			headB = headB.next;
		}
		return headA == null ? null : headA;
	}

	public int getLength(ListNode head) {
		int len = 0;
		ListNode cur = head;
		while (cur != null) {
			len++;
			cur = cur.next;
		}
		return len;
	}

	/*
	 * http://www.cnblogs.com/theskulls/p/4881142.html
	 */
	public long permutationIndex(int[] A) {
		// Write your code here
		int n = A.length;
		long fact = 1;
		long index = 0;
		for (int i = n - 2; i >= 0; i--) {
			int bigger = 0;
			for (int j = i + 1; j < n; j++) {
				if (A[j] < A[i])
					bigger++;
			}
			index += bigger * fact;
			fact *= n - i;
		}
		return index + 1;
	}


	/*
	 * Given a permutation which may contain repeated numbers, find its index in
	 * all the permutations of these numbers, which are ordered in
	 * lexicographical order. The index begins at 1.
	 */
	public long permutationIndexII(int[] A) {
		// Write your code here
		long index = 0;
		long fact = 1;
		long multifact = 1;
		int n = A.length;
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();

		for (int i = n - 1; i >= 0; i--) {
			if (map.containsKey(A[i])) {
				map.put(A[i], map.get(A[i]) + 1);
				multifact *= map.get(A[i]);
			} else {
				map.put(A[i], 1);
			}
			int rank = 0;
			for (int j = i + 1; j < A.length; j++) {
				if (A[i] > A[j])
					rank++;
			}

			index += rank * fact / multifact;
			fact *= n - i;
		}
		return index + 1;
	}
	
	public int kthSmallest(int[][] matrix, int k) {
        // write your code here
        int n=matrix.length;
        int m=matrix[0].length;
        PriorityQueue<Node> heap=new PriorityQueue<Node>(n, new Comparator<Node>(){
            @Override
            public int compare(Node n1, Node n2){
                return n1.val-n2.val;
            }
            });
        for(int i=0;i<n;i++){
            heap.offer(new Node(matrix[i][0], i, 0));
        }
        int res=0;
        while(k>0){
            Node cur=heap.poll();
            res=cur.val;
            int row=cur.row;
            int col=cur.col;
            if(col+1<m)
                heap.offer(new Node(matrix[row][col+1], row, col+1));
            k--;
        }
        return res;
    }
    
    class Node{
        int row;
        int col;
        int val;
        
        public Node(int val, int row, int col){
            this.row=row;
            this.col=col;
            this.val=val;
        }
    }
    
    public int divide(int dividend, int divisor) {
        // Write your code here
        boolean neg=false;
        boolean overflow=false;
        if(dividend<0&&divisor>0||dividend>0&&divisor<0)
            neg=true;
        long dvd=Math.abs((long)dividend);
        long dvs=Math.abs((long)divisor);
        
        int res=0;
        while(dvd>=dvs){
            int shift=0;
            while(dvd>=(dvs<<shift)){
                shift++;
            }
            shift--;
            dvd-=dvs<<shift;
            if(Integer.MAX_VALUE-(1<<shift)>res)
                res+=(1<<shift);
            else{
                overflow=true;
                break;
            }
            
        }
        if(overflow){
            return neg? Integer.MIN_VALUE: Integer.MAX_VALUE;
        }
        return neg? -res: res;
    }
    
    public String getPermutation(int n, int k) {
        List<Integer> nums=new ArrayList<Integer>();
        int fact=1;
        for(int i=1;i<=n;i++){
            nums.add(i);
            fact*=i;
        }
        k--;
        
        StringBuilder sb=new StringBuilder();
        for(int i=0;i<n;i++){
            fact/=(n-i);
            int index=k/fact;
            sb.append(nums.get(index));
            k%=fact;
            nums.remove(index);
        }
        return sb.toString();
    }

	// public boolean isAdditiveNumber(String num) {
	// if(num==null||num.length()<3)
	// return false;
	//
	// }

	// public int trapRainWater(int[][] heights) {
	// // write your code here
	// int m=heights.length;
	// if(m<3)
	// return 0;
	// int n=heights[0].length;
	// if(n<3)
	// return 0;
	// boolean[][] visited=new boolean[m][n];
	//
	// for(int i=1;i<m;i++){
	// for(int j=1;j<n;j++){
	// if(!visited[i][j]&&heights[i][j])
	// }
	// }
	//
	// }

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Solutions sol = new Solutions();
		int[] A = { 1, 3, 4, 5, 8, 10, 11, 12, 14, 17, 20, 22, 24, 25, 28, 30,
				31, 34, 35, 37, 38, 40, 42, 44, 45, 48, 51, 54, 56, 59, 60, 61,
				63, 66 };
		System.out.println(sol.kSum1(A, 24, 842));
		System.out.println(sol.kSum2(A, 24, 842));

		int[] num = { 2, 3, 6, 7 };
		System.out.println(sol.combinationSum2(num, 7));

		Set<String> dict = new HashSet<String>();
		dict.add("a");

		System.out.println(sol.wordSegmentation2("a", dict));

		System.out.println(sol.bitSwapRequired(1, -1));
		System.out.println(sol.DeleteDigits("178542", 4));
		System.out.println(sol.DeleteDigits("123456789", 1));

		int[] input = { 26, -31, 10, -29, 17, 18, -24, -10 };
		sol.rerange(input);

		int[] jumps = { 17, 8, 29, 17, 35, 28, 14, 2, 45, 8, 6, 54, 24, 43, 20,
				14, 33, 31, 27, 11 };
		System.out.println(sol.jump(jumps));

		int[] nums = { -8, 0, 1, 2, 3 };
		System.out.println(sol.maxTwoSubArrays(nums));
		System.out.println(sol.kthPrimeNumber(20));

		System.out.println(sol.median(nums));

		List<ListNode> lists = new ArrayList<ListNode>();
		lists.add(null);

		sol.mergeKLists(lists);

		ArrayList<Integer> arr = new ArrayList<Integer>();
		arr.add(1);
		arr.add(2);
		arr.add(3);

		System.out.println(sol.productExcludeItself(arr));

		System.out.println(sol.removeDuplicates2(nums));
		int[][] matrix = { { 1, 3, 5, 7 }, { 10, 11, 16, 20 },
				{ 23, 30, 34, 50 } };
		System.out.println(sol.searchMatrix2(matrix, 7));

		char[][] board = { { 'b', 'b', 'a', 'a', 'b', 'a' },
				{ 'b', 'b', 'a', 'b', 'a', 'a' },
				{ 'b', 'b', 'b', 'b', 'b', 'b' },
				{ 'a', 'a', 'a', 'b', 'a', 'a' },
				{ 'a', 'b', 'a', 'a', 'b', 'b' } };
		ArrayList<String> words = new ArrayList<String>();
		words.add("abbbababaa");
		words.add("bbb");
		words.add("aba");
		words.add("ba");
		words.add("ab");
		words.add("c");
		words.add("dba");
		words.add("bad");

		System.out.println(sol.wordSearchII2(board, words));

		int[] colors = { 2, 1, 1, 2, 2 };
		sol.sortColors2(colors, 2);

		System.out.println(sol.minWindow("abc", "ac"));

		System.out.println(sol.minCut("ab"));

		int[] A1 = { 1, 2, 3, 4, 5, 6 }, B = { 2, 3, 4, 5 };
		System.out.println(sol.findMedianSortedArrays(A1, B));

		ListNode n1 = new ListNode(29);
		n1.next = new ListNode(5);
		ListNode[] hashtable = { null, null, n1 };

		sol.rehashing(hashtable);
		System.out.println(sol.solveNQueens(4));

		System.out.println(sol.updateBits(1024, 21, 2, 6));

		int[] testcase = { 1, 2, 7, 7, 8 };
		Interval i1 = new Interval(1, 2);
		Interval i2 = new Interval(0, 4);
		Interval i3 = new Interval(2, 4);
		ArrayList<Interval> queries = new ArrayList<Interval>();
		queries.add(i1);
		queries.add(i2);
		queries.add(i3);
		SegmentTreeNode sroot = sol.build2(testcase, 0, 4);
		sol.inorder(sroot);
		System.out.println();
		System.out.println(sol.intervalSum(testcase, queries));
		System.out.println(sol.maxSlidingWindow(testcase, 1));

		int[][] zmatrix = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } };
		sol.printZMatrix2(zmatrix);
		System.out.println(Arrays.toString(sol.printZMatrix(zmatrix)));

		ListNode tnode = new ListNode(1);
		tnode.next = new ListNode(1);
		sol.insertionSortList(tnode);

		String[] expression = { "2", "*", "6", "-", "(", "23", "+", "7", ")",
				"/", "(", "1", "+", "2", ")" };
		String[] expression1 = { "(", "(", "(", "(", "(", ")", ")", ")", ")",
				")" };
		System.out.println(sol.evaluateExpression(expression1));

		System.out.println(sol.lengthOfLongestSubstringKDistinct(
				"eqgkcwGFvjjmxutystqdfhuMblWbylgjxsxgnoh", 16));

		ListNode l = new ListNode(1);
		l.next = new ListNode(2);
		l.next.next = new ListNode(3);
		l.next.next.next = new ListNode(4);

		sol.deleteMiddleNode(l);
		while (l != null) {
			System.out.print(l.val + " ");
			l = l.next;
		}
		String w = "hello world";
		char[] w2c = w.toCharArray();
		// System.out.println(sol.replaceBlank(w2c, 11));

		System.out.println(sol.simplifyPath("/a/./b/../../c/"));
		System.out.println(sol.countAndSay(4));
		System.out.println(sol.addDigits(19));
		System.out.println(sol.longestPalindrome("bb"));

		System.out.println(sol.restoreIpAddresses("0000"));

		int[][] ma = { { 1, 5, 7 }, { 3, 7, -8 }, { 4, -8, 9 } };
		sol.submatrixSum(ma);

	}

}
