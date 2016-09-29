package com.google;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
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

public class GoogleSolutions {

	public String reArrange(String s, int k) throws Exception {
		Map<Character, Integer> map = new HashMap<Character, Integer>();
		for (char c : s.toCharArray()) {
			if (!map.containsKey(c))
				map.put(c, 1);
			else
				map.put(c, map.get(c) + 1);
		}

		PriorityQueue<Node> heap = new PriorityQueue<Node>(map.size(),
				new Comparator<Node>() {

					@Override
					public int compare(Node o1, Node o2) {
						if (o1.freq != o2.freq)
							return o2.freq - o1.freq;
						return o1.c - o2.c;
					}

				});

		for (char c : map.keySet()) {
			heap.offer(new Node(c, map.get(c)));
		}

		StringBuilder sb = new StringBuilder();
		Queue<Node> que = new LinkedList<Node>();
		for (int cur = 0; cur < s.length(); cur++) {
			if (!que.isEmpty() && que.peek().lastPos + k <= cur) {// k dist:
																	// cur-lastpost>=k
				heap.offer(que.poll());
			}
			if (heap.isEmpty())
				throw new Exception("invalid input");

			Node node = heap.poll();
			sb.append(node.c);
			node.lastPos = cur;
			node.freq--;
			if (node.freq != 0)
				que.offer(node);
		}
		return sb.toString();
	}

	/*
	 * Given: int[] F of size k, with numbers in [0, k) int a_init, within [0,k)
	 * int N A_0 = a_init A_1 = F[A_0] A_2 = F[A_1] ... A_i = F[A_i-1]
	 * 
	 * Find A_N.
	 */

	public int findNumber(int[] F, int n, int start) {
		if (n == 0)
			return start;
		int res = start;
		for (int i = 1; i <= n; i++) {
			res = F[res];
		}
		return res;
	}

	public int findNumber2(int[] F, int n, int start) {
		if (n == 0)
			return start;
		int[] A = new int[n + 1];
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		int cycle = -1;
		A[0] = start;
		for (int i = 1; i <= n; i++) {
			A[i] = F[A[i - 1]];
			if (!map.containsKey(A[i])) {
				map.put(A[i], i);
			} else {
				cycle = i - map.get(A[i]);
				return A[(n - i) % cycle + map.get(A[i])];
			}
		}
		return A[n];
	}

	/*
	 * 给一个手机键盘，input是一个string，把input根据键盘转化成数字，比如input是“RAT”,
	 * output就应该是“77728”，因为R要按三次7键。写之前也是确定了一下input会不会有不合法情况
	 */

	public String convert2Num(String s) {
		if (s.isEmpty())
			return "";
		String[] strs = { "", "", "ABC", "DEF", "GHI", "JKL", "MNO", "PQRS",
				"TUV", "WXYZ" };
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			for (int j = 0; j < strs.length; j++) {
				int index = strs[j].indexOf(c);
				if (index != -1) {
					for (int k = 0; k <= index; k++) {
						sb.append(j);
					}
				}
			}
		}
		return sb.toString();
	}

	// Returns true if arr[] can be partitioned in two
	// subsets of equal sum, otherwise false
	public boolean isSubsetSum(int arr[]) {
		int n = arr.length;
		int sum = 0;
		for (int num : arr) {
			sum += num;
		}
		if (sum % 2 != 0)
			return false;
		// part[i][j] = true if a subset of {arr[0], arr[1], ..arr[j-1]} has sum
		// equal to i, otherwise false
		boolean[][] dp = new boolean[n + 1][sum / 2 + 1];

		for (int i = 0; i < sum / 2 + 1; i++) {
			dp[0][i] = false;
		}
		for (int i = 0; i <= n; i++) {
			dp[i][0] = true;
		}

		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= sum / 2; j++) {
				dp[i][j] = dp[i - 1][j];
				if (j >= arr[i - 1]) {
					dp[i][j] = dp[i][j] || dp[i - 1][j - arr[i - 1]];
				}
			}
		}
		return dp[n][sum / 2];
	}

	public int subsetSumWays(int arr[]) {
		int n = arr.length;
		int sum = 0;
		for (int num : arr) {
			sum += num;
		}
		if (sum % 2 != 0)
			return 0;

		int[] dp = new int[sum / 2 + 1];
		dp[0] = 1;

		for (int i = 0; i < n; i++) {
			for (int j = sum / 2; j >= arr[i]; j--) {
				dp[j] += dp[j - arr[i]];
			}
		}
		return dp[sum / 2];
	}

	private static int getmNumberOfSubsets(int[] numbers, int sum) {
		int[] dp = new int[sum + 1];
		dp[0] = 1;
		int currentSum = 0;
		for (int i = 0; i < numbers.length; i++) {
			currentSum += numbers[i];
			for (int j = Math.min(sum, currentSum); j >= numbers[i]; j--)
				dp[j] += dp[j - numbers[i]];
		}

		return dp[sum];
	}

	public List<List<Integer>> combinationSum(int[] A, int target) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		Arrays.sort(A);
		combinationSumUtil(A, target, sol, res, 0, 0);
		return res;
	}

	public void combinationSumUtil(int[] A, int target, List<Integer> sol,
			List<List<Integer>> res, int cur, int cursum) {
		if (cursum > target || cur > A.length)
			return;
		if (cursum == target) {
			res.add(new ArrayList<Integer>(sol));
			return;
		}

		for (int i = cur; i < A.length; i++) {
			cursum += A[i];
			sol.add(A[i]);
			combinationSumUtil(A, target, sol, res, i + 1, cursum);
			cursum -= A[i];
			sol.remove(sol.size() - 1);
		}
	}

	public String encode(List<String> strs) {
		StringBuilder sb = new StringBuilder();
		for (String s : strs) {
			int len = s.length();
			sb.append(len).append("/").append(s);
		}
		return sb.toString();
	}

	public List<String> decode(String s) {
		List<String> res = new ArrayList<String>();
		int i = 0;
		while (i < s.length()) {
			int slash = s.indexOf('/', i);
			int size = Integer.valueOf(s.substring(i, slash));
			String str = s.substring(slash + 1, slash + 1 + size);
			res.add(str);
			i = slash + 1 + size;
		}
		return res;
	}

	public String serialize(TreeNode root) {
		if (root == null)
			return "";
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

	public TreeNode deserialize(String s) {
		String[] strs = s.split(",");
		Queue<String> que = new LinkedList<String>(Arrays.asList(strs));
		return deserialize(que);

	}

	public TreeNode deserialize(Queue<String> que) {
		String val = que.poll();
		if (val.equals("#"))
			return null;
		TreeNode root = new TreeNode(Integer.valueOf(val));
		root.left = deserialize(que);
		root.right = deserialize(que);
		return root;
	}

	// serialzie Nary tree
	public String serialize(NaryNode root) {
		StringBuilder sb = new StringBuilder();
		serialize(root, sb);
		return sb.toString();
	}

	public void serialize(NaryNode root, StringBuilder sb) {
		if (root == null) {
			return;
		}
		sb.append(root.val + ",");

		for (NaryNode child : root.children) {
			serialize(child, sb);
		}

		sb.append("$,");
	}

	// deserialize nary tree
	public NaryNode deSerialize(String s) {
		String[] ss = s.split(",");
		Queue<String> que = new LinkedList<String>(Arrays.asList(ss));
		return buildTree(que);
	}

	public NaryNode buildTree(Queue<String> que) {
		String val = que.poll();
		if (val.equals("$"))
			return null;
		NaryNode root = new NaryNode(Integer.valueOf(val));

		while (true) {
			NaryNode child = buildTree(que);
			if (child == null)
				break;
			root.children.add(child);
		}
		return root;
	}

	public List<List<Integer>> printTree(NaryNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (root == null)
			return res;
		Queue<NaryNode> que = new LinkedList<NaryNode>();
		List<Integer> level = new ArrayList<Integer>();
		int curlevel = 0, nextlevel = 0;
		que.add(root);
		curlevel++;
		while (!que.isEmpty()) {
			NaryNode top = que.remove();
			level.add(top.val);
			curlevel--;
			if (top.children != null) {
				for (NaryNode node : top.children) {
					que.add(node);
					nextlevel++;
				}
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

	// 给一堆votes(candidate, timestamp)，问当前时刻T得票最高的人是谁。Follow up问得票最高的前K个人。

	public String highestVotes(List<Vote> votes, int T) {
		Map<String, Integer> count = new HashMap<String, Integer>();
		int max = 0;
		String res = "";
		for (Vote vote : votes) {
			if (vote.time <= T) {
				String name = vote.name;
				if (count.containsKey(name)) {
					count.put(name, count.get(name) + 1);
				} else {
					count.put(name, 1);
				}
				if (count.get(name) > max) {
					max = count.get(name);
					res = name;
				}
			}
		}
		return res;
	}

	public List<String> highestKVotes(List<Vote> votes, int T, int k) {
		Map<String, Integer> count = new HashMap<String, Integer>();
		for (Vote vote : votes) {
			if (vote.time <= T) {
				String name = vote.name;
				if (count.containsKey(name)) {
					count.put(name, count.get(name) + 1);
				} else {
					count.put(name, 1);
				}
			}
		}

		PriorityQueue<VoteNode> heap = new PriorityQueue<VoteNode>(k,
				new Comparator<VoteNode>() {
					@Override
					public int compare(VoteNode v1, VoteNode v2) {
						return v1.num - v2.num;
					}
				});
		for (String name : count.keySet()) {
			VoteNode node = new VoteNode(name, count.get(name));
			if (heap.size() < k) {
				heap.offer(node);
			} else {
				VoteNode top = heap.peek();
				if (top.num < node.num) {
					heap.poll();
					heap.offer(node);
				}
			}
		}
		List<String> res = new ArrayList<String>();
		while (!heap.isEmpty()) {
			res.add(heap.poll().name);
		}
		return res;
	}

	// 给你一个数字 拆成个位数相乘 一直重复 知道那个数不能再拆

	public int superHappyNum(int num) {
		while (num >= 10) {
			int t = num;
			int cur = 1;
			while (t > 0) {
				cur *= t % 10;
				t /= 10;
			}
			num = cur;
		}
		return num;
	}

	/*
	 * Give a stream of numbers one at time. Compute the average of the last N
	 * numbers.
	 * 
	 * Input: 1, 2, 3. Give N=2. Answer: 1, 1.5, 2.5.
	 */

	public List<Double> windowAverage(List<Integer> nums, int N) {
		List<Double> res = new ArrayList<Double>();
		Queue<Integer> que = new LinkedList<Integer>();
		int curSum = 0;
		for (int i = 0; i < nums.size(); i++) {
			if (que.size() < N) {
				que.offer(nums.get(i));
				curSum += nums.get(i);
			} else {
				curSum -= que.poll();
				que.offer(nums.get(i));
				curSum += nums.get(i);
			}
			res.add(1.0 * curSum / que.size());
		}
		return res;
	}

	// 求两个最大公约数
	public int greatestCommonDivisor(int n, int m) {
		if (n < m) {
			int t = n;
			n = m;
			m = t;
		}

		while (n % m != 0) {
			int rem = n % m;
			n = m;
			m = rem;
		}
		return m;
	}

	// 多个数的GCD
	public int greatestCommonDivisisor(int[] nums) {
		int res = nums[0];
		for (int i = 1; i < nums.length; i++) {
			res = greatestCommonDivisor(res, nums[i]);
		}
		return res;
	}

	// 最小公倍数 least common multiple
	public int leastCommonMultiple(int a, int b) {
		return a * b / greatestCommonDivisor(a, b);
	}

	// 多个数的lcm
	public int leasetCommonMultiple(int[] nums) {
		int lcm = nums[0];
		for (int i = 1; i < nums.length; i++) {
			lcm = leastCommonMultiple(lcm, nums[i]);
		}
		return lcm;
	}

	// n个人出去游玩 其中几个人付了钱 回来的时候要分账 求最少的transactions的算法
	// interviewee提示说最后可以用类似2-sum，3-sum的方法做
	/*
	 * solution: 把每个人欠的钱组成一个数列，所有人欠的钱和多付钱的人之和自然为0，先找出所有sum为0的数对，除去这些数对，
	 * 剩下的数组长度减一就是transaction的最小值 .
	 * 比如-4，-3，-1，1，1，2，4，先找出sum为0的数对（-4，4），（-1，1），除去这两对，数组剩下-3，1，2
	 * transaction最小值 = 2（sum为0的数对数）+ 2（数组长度减一）= 4
	 */
	public int fewestTransactions(int[] fees) {
		if (fees.length < 2)
			return 0;
		int totalPay = 0;
		for (int fee : fees) {
			totalPay += fee;
		}

		int subFee = totalPay / fees.length;

		for (int i = 0; i < fees.length; i++) {
			fees[i] = fees[i] - subFee;
		}
		Arrays.sort(fees);
		int n = fees.length;
		int transaction = 0;
		int i = 0, j = n - 1;
		while (i < j) {
			if (fees[i] == 0) {
				n--;
				i++;
			}
			if (fees[j] == 0) {
				n--;
				j--;
			}
			if (fees[i] + fees[j] == 0) {
				transaction++;
				i++;
				j--;
				n -= 2;
			} else if (fees[i] + fees[j] > 0)
				j--;
			else
				i++;
		}
		return transaction + n - 1;
	}

	/*
	 * given grid of colors, coordinate of a point and its color, find the
	 * perimeter of the region that has the same color of that point.
	 * BFS或DFS，构成perimeter的条件是只要上下左右有一个不是同颜色或者是out of bound 用一个set记录visit的信息
	 */
	private final int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };

	public int findPerimeter(int[][] mat, int r, int c) {
		int m = mat.length, n = mat[0].length;
		boolean[][] visited = new boolean[m][n];
		visited[r][c] = true;
		Queue<Point> que = new LinkedList<Point>();
		que.offer(new Point(r, c));
		int perimeter = 4;
		while (!que.isEmpty()) {
			Point p = que.poll();
			for (int[] dir : dirs) {
				int i = p.x + dir[0], j = p.y + dir[1];
				if (i < 0 || j < 0 || i >= m || j >= n
						|| mat[i][j] != mat[p.x][p.y])
					continue;
				else {
					if (!visited[i][j]) {
						que.offer(new Point(i, j));
						visited[i][j] = true;
						int count = 0;
						for (int[] d : dirs) {
							int row = i + d[0], col = j + d[1];
							if (row < 0 || col < 0 || row >= m || col >= n)
								continue;
							if (visited[row][col])
								count++;
						}
						if (count == 1)
							perimeter += 2;
						else if (count == 3)
							perimeter -= 2;
						else if (count == 4)
							perimeter -= 4;
					}
				}

			}
		}
		return perimeter;
	}

	// 给出一个 list of int, the target, 输出这个 list 中所用的数能否通过4则运算 得到 target。
	public boolean evaluatesTo(List<Integer> nums, int target) {
		if (nums.size() == 0)
			return false;
		Set<Double> res = evaluatesToDFS(0, nums.size() - 1, nums);
		System.out.println(res);
		System.out.println(res.size());
		for (double i : res) {
			if (i == target)
				return true;
		}
		return false;
	}

	public Set<Double> evaluatesToDFS(int left, int right, List<Integer> nums) {
		Set<Double> res = new HashSet<Double>();
		if (left > right)
			return res;
		if (left == right) {
			res.add(1.0 * nums.get(left));
			return res;
		}

		for (int i = left; i < right; i++) {
			Set<Double> leftSol = evaluatesToDFS(left, i, nums);
			Set<Double> rightSol = evaluatesToDFS(i + 1, right, nums);

			for (double a : leftSol) {
				for (double b : rightSol) {
					res.add(a + b);
					res.add(a - b);
					res.add(a * b);
					if (b != 0)
						res.add(a / b);
				}
			}
		}
		return res;
	}

	// 给出一个 list of int, the target, 输出这个 list 中所用的数能否通过4则运算 得到 target。
	public List<String> addOperators(int[] nums, int target) {
		List<String> res = new ArrayList<String>();
		if (nums.length == 0)
			return res;
		addOperatorsUtil(0, nums, target, "", res, 0, 0, 0);
		return res;
	}

	public void addOperatorsUtil(int dep, int[] nums, int target, String sol,
			List<String> res, int curPos, double curSum, double prevNum) {
		if (dep == nums.length && curSum == target) {
			res.add(sol);
			return;
		}

		for (int i = curPos; i < nums.length; i++) {
			if (i == 0) {
				addOperatorsUtil(dep + 1, nums, target, sol + nums[i], res,
						i + 1, curSum + nums[i], nums[i]);
			} else {
				addOperatorsUtil(dep + 1, nums, target, sol + "+" + nums[i],
						res, i + 1, curSum + nums[i], nums[i]);
				addOperatorsUtil(dep + 1, nums, target, sol + "-" + nums[i],
						res, i + 1, curSum - nums[i], -nums[i]);
				addOperatorsUtil(dep + 1, nums, target, sol + "*" + nums[i],
						res, i + 1, curSum - prevNum + prevNum * nums[i],
						prevNum * nums[i]);
				if (nums[i] != 0)
					addOperatorsUtil(dep + 1, nums, target,
							sol + "/" + nums[i], res, i + 1, curSum - prevNum
									+ prevNum / nums[i], prevNum * nums[i]);
			}
		}
	}

	// Longest Alternating/zigzag Subsequence
	public int longestZigzagSequence(int[] nums) {
		int n = nums.length;
		// Length of the longest ZigZag subsequence ending at index i and last
		// element is less than its previous element
		int[] dp1 = new int[n];
		// Length of the longest ZigZag subsequence ending at index i and last
		// element is greater than its previous element
		int[] dp2 = new int[n];
		Arrays.fill(dp1, 1);
		Arrays.fill(dp2, 1);

		int max = 1;
		for (int i = 1; i < n; i++) {
			for (int j = 0; j < i; j++) {
				if (nums[i] > nums[j] && dp1[i] < dp2[j] + 1)
					dp1[i] = dp2[j] + 1;
				if (nums[i] < nums[j] && dp2[i] < dp1[j] + 1)
					dp2[i] = dp1[j] + 1;
			}
			max = Math.max(dp1[i], dp2[i]);
		}
		return max;
	}

	// 给了一堆点（x_0,y_0)....(x_n,y_n)问怎么判断是不是关于任意vertical line 轴对称
	public boolean axialSymmetric(Point[] points) {
		Map<Integer, List<Integer>> map = new HashMap<Integer, List<Integer>>();
		int sum = 0;
		for (Point p : points) {
			int x = p.x, y = p.y;
			sum += x;
			if (!map.containsKey(y)) {
				List<Integer> lst = new ArrayList<Integer>();
				lst.add(x);
				map.put(y, lst);
			} else {
				map.get(y).add(x);
			}
		}
		double mean = sum / points.length;
		for (int y : map.keySet()) {
			double m = getMean(map.get(y));
			if (m != mean)
				return false;
		}
		return true;
	}

	/*
	 * 输入是一个int的array和size, 两个array,
	 * 一个是从头加到当前的和，就是sum=nums[0]+nums[1]+...+nums,另一个是从尾巴开始加
	 * 。用这俩找到一个index能把输入的array分成两部分让两边的和的差最小，返回index
	 */

	public int minDifference(int[] nums) {
		int preSum = 0, suffSum = 0;
		int i = 0, j = nums.length - 1;

		while (i < j) {
			if (preSum < suffSum) {
				preSum += nums[i++];
			} else {
				suffSum += nums[j--];
			}
		}
		return i;
	}

	public double getMean(List<Integer> lst) {
		int sum = 0;
		for (int y : lst) {
			sum += y;
		}
		return 1.0 * sum / lst.size();
	}

	/*
	 * Suppose you have a list of prime numbers in order, but not consecutive.
	 * For example [3,7,13] Find the N smallest integers in order which have
	 * only these primes as factors. For example [3,7,9,13,21,...
	 */

	public int[] firstNSuperUglyNumbers(int[] primes, int n) {
		int[] res = new int[n + 1];
		res[0] = 1;
		int[] idxs = new int[primes.length];
		for (int i = 1; i <= n; i++) {
			int min = Integer.MAX_VALUE;
			for (int j = 0; j < primes.length; j++) {
				min = Math.min(min, res[idxs[j]] * primes[j]);
			}
			res[i] = min;
			for (int j = 0; j < primes.length; j++) {
				if (min % primes[j] == 0)
					idxs[j]++;
			}
		}
		return Arrays.copyOfRange(res, 1, res.length);
	}

	// 1. 寻找最长递增数列，init是找长度，follow up是打印出序列，follow follow up是打印出所有可能序列，follow
	// follow follow up 是变成字符串找最长递增。
	public int longestIncreasingSequence(int[] A) {
		int[] dp = new int[A.length];
		Arrays.fill(dp, 1);
		int max = 1;
		for (int i = 1; i < A.length; i++) {
			for (int j = 0; j < i; j++) {
				if (A[i] > A[j] && dp[i] < dp[j] + 1) {
					dp[i] = dp[j] + 1;
					max = Math.max(max, dp[i]);
				}
			}
		}
		return max;
	}

	public List<Integer> printLongestIncreasingSequence(int[] A) {
		List<Integer> res = new ArrayList<Integer>();
		int[] dp = new int[A.length];
		Arrays.fill(dp, 1);
		int max = 1;
		for (int i = 1; i < A.length; i++) {
			for (int j = 0; j < i; j++) {
				if (A[i] > A[j] && dp[i] < dp[j] + 1) {
					dp[i] = dp[j] + 1;
					max = Math.max(max, dp[i]);
				}
			}
		}

		System.out.println(Arrays.toString(dp));
		int cur = max;
		for (int i = A.length - 1; i >= 0; i--) {
			if (dp[i] == cur) {
				res.add(0, A[i]);
				cur--;
			}
		}
		return res;
	}

	public List<String> printAllPossibleLIS(int[] A) {
		List<String> res = new ArrayList<String>();
		List<String>[] path = new ArrayList[A.length];
		int[] dp = new int[A.length];
		Arrays.fill(dp, 1);
		for (int i = 0; i < A.length; i++) {
			List<String> l = new ArrayList<String>();
			l.add("" + A[i]);
			path[i] = l;
		}
		int max = 1;
		for (int i = 1; i < A.length; i++) {
			for (int j = 0; j < i; j++) {
				if (A[i] > A[j] && dp[i] <= dp[j] + 1) {
					if (dp[i] == dp[j] + 1) {
						for (String s : path[j]) {
							path[i].add(s + " " + A[i]);
						}
					} else {

						dp[i] = dp[j] + 1;
						max = Math.max(max, dp[i]);
						List<String> lst = new ArrayList<String>();
						for (String s : path[j]) {
							lst.add(s + " " + A[i]);
						}
						path[i] = lst;
					}
				}
			}
		}

		for (int i = 0; i < path.length; i++) {
			for (String s : path[i]) {
				if (s.split(" ").length == max) {
					res.add(s);
				}

			}
		}
		return res;
	}

	/*
	 * 给两个string,其中一个string比另外一个多了个字母，返回这个字母 已按顺序
	 */
	public char findExtraCharacter(String s1, String s2) {
		if (s1.length() > s2.length()) {
			String s = s1;
			s1 = s2;
			s2 = s;
		}

		int i = 0;
		while (i < s1.length()) {
			if (s1.charAt(i) != s2.charAt(i)) {
				if (s1.charAt(i) == s2.charAt(i + 1))
					return s2.charAt(i);
				else
					return s2.charAt(i + 1);
			} else
				i++;
		}
		return s2.charAt(i);
	}

	class HeapNode {
		int val; // element to be sorted
		int i; // index of array from which element picked
		int j; // index of next element

		public HeapNode(int val, int i, int j) {
			this.val = val;
			this.i = i;
			this.j = j;
		}
	}

	public int[] mergeKsortedArrays(int[][] arr, int k) {
		int n = arr[0].length;
		int[] res = new int[n * k];
		PriorityQueue<HeapNode> heap = new PriorityQueue<HeapNode>(k,
				new Comparator<HeapNode>() {

					@Override
					public int compare(HeapNode o1, HeapNode o2) {
						// TODO Auto-generated method stub
						return o1.val - o2.val;
					}

				});

		for (int i = 0; i < k; i++) {
			heap.offer(new HeapNode(arr[i][0], i, 1));
		}
		int idx = 0;

		while (!heap.isEmpty()) {
			HeapNode node = heap.poll();
			res[idx++] = node.val;
			if (node.j < arr[node.i].length) {
				heap.offer(new HeapNode(arr[node.i][node.j], node.i, node.j + 1));
			}
		}
		return res;
	}

	/*
	 * * Finds first non repeated character in a String in just one pass. * It
	 * uses two storage to cut down one iteration, standard space vs time *
	 * trade-off.Since we store repeated and non-repeated character separately,
	 * * at the end of iteration, first element from List is our first non *
	 * repeated character from String.
	 */

	public char firstNonRepeatingChar(String word) {
		Set<Character> repeatingChars = new HashSet<Character>();
		List<Character> nonRepeated = new ArrayList<Character>();
		for (char c : word.toCharArray()) {
			if (repeatingChars.contains(c))
				continue;
			if (nonRepeated.contains(c)) {
				repeatingChars.add(c);
				nonRepeated.remove((Character) c);
			} else
				nonRepeated.add(c);
		}
		return nonRepeated.size() > 0 ? nonRepeated.get(0) : null;
	}

	/*
	 * Multiply two polynomials Given two polynomials represented by two arrays,
	 * write a function that multiplies given two polynomials. Input: A[] = {5,
	 * 0, 10, 6} B[] = {1, 2, 4} Output: prod[] = {5, 10, 30, 26, 52, 24}
	 * 
	 * The first input array represents "5 + 0x^1 + 10x^2 + 6x^3" The second
	 * array represents "1 + 2x^1 + 4x^2" And Output is
	 * "5 + 10x^1 + 30x^2 + 26x^3 + 52x^4 + 24x^5"
	 */

	public void multiply(int[] A, int[] B) {
		int m = A.length, n = B.length;
		int[] res = new int[m + n - 1];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				res[i + j] += A[i] * B[j];
			}
		}

		for (int i = 0; i < res.length; i++) {
			if (res[i] == 0)
				continue;
			System.out.print(res[i]);
			if (i != 0)
				System.out.print("x^" + i);
			if (i != res.length - 1)
				System.out.print("+");
		}
		System.out.println();
	}

	/*
	 * 一个字典，里面很多单词，例如 google, leg, about, lemma, apple, time 找这样的pair <A,
	 * B>，有两个条件, (1) A单词的后两个字母和B单词的前两个字母一样 （2）A单词的第一个字母和B单词的最后一个字母一样， 例如<google,
	 * leg>就是一个合格的pair，<apple, lemma>也是一个合格的pair， <about, time>不可以
	 * 然后求这样的pair的最长长度，<apple, lemma>的长度=5+5=10
	 */

	public int longestPair(List<String> words) {
		Map<String, String> map1 = new HashMap<String, String>();
		Map<String, String> map2 = new HashMap<String, String>();
		int max = 0;
		String word1 = "", word2 = "";
		for (String word : words) {
			String key1 = word.substring(word.length() - 2) + word.charAt(0);
			if (!map1.containsKey(key1)
					|| map1.get(key1).length() < word.length())
				map1.put(key1, word);
			String key2 = word.substring(0, 2) + word.charAt(word.length() - 1);
			if (!map2.containsKey(key1)
					|| map2.get(key2).length() < word.length())
				map2.put(key2, word);
			if (map1.containsKey(key2)) {
				if (word.length() + map1.get(key2).length() > max) {
					max = word.length() + map1.get(key2).length();
					word1 = word;
					word2 = map1.get(key2);
				}

			}
			if (map2.containsKey(key1)) {
				if (word.length() + map2.get(key1).length() > max) {
					max = word.length() + map2.get(key1).length();
					word1 = word;
					word2 = map2.get(key1);
				}
			}
		}
		System.out.println(map1);
		System.out.println(map2);
		System.out.println(word1 + " " + word2);
		return max;
	}

	/*
	 * 一个string, 有空格，有引号，sparse string, 两个引号中间的部分的空格不处理, 引号外面的空格将前后划成两个String
	 */
	public List<String> parseString(String s) {
		List<String> res = new ArrayList<String>();
		int start = 0, end = 0;
		boolean quote = false;
		while (end < s.length()) {
			if (s.charAt(end) == ' ' && !quote) {
				if (!s.substring(start, end).isEmpty()) {
					res.add(s.substring(start, end));
				}
				start = end + 1;
			} else if (s.charAt(end) == '"') {
				quote = !quote;
				if (quote)
					start++;
				else {
					res.add(s.substring(start, end));
					start = end + 1;
				}
			}
			end++;
		}
		res.add(s.substring(start, end));
		return res;
	}

	/*
	 * Minimum insertions to form a palindrome
	 */

	public int findMinInsertions(String s) {
		return findMinInsertions(s, 0, s.length() - 1);
	}

	public int findMinInsertions(String s, int l, int r) {
		if (l > r)
			return Integer.MAX_VALUE;
		if (l == r)
			return 0;
		if (l == r - 1)
			return s.charAt(l) == s.charAt(r) ? 0 : 1;
		return s.charAt(l) == s.charAt(r) ? findMinInsertions(s, l + 1, r - 1)
				: Math.min(findMinInsertions(s, l + 1, r),
						findMinInsertions(s, l, r - 1)) + 1;
	}

	// /dp[i][j] will store
	// minumum number of insertions needed to convert str[i..j]
	// to a palindrome.
	public int findMinInsertionsDP(String s) {
		int n = s.length();
		int[][] dp = new int[n][n];

		for (int gap = 1; gap < n; gap++) {
			for (int i = 0; i + gap < n; i++) {
				if (s.charAt(i) == s.charAt(i + gap))
					dp[i][i + gap] = dp[i + 1][i + gap - 1];
				else
					dp[i][i + gap] = Math.min(dp[i][i + gap - 1], dp[i + 1][i
							+ gap]) + 1;
			}
		}
		return dp[0][n - 1];
	}

	public int findMinInsertionsDP2(String s) {
		int n = s.length();
		int[][] dp = new int[n][n];

		for (int i = n - 2; i >= 0; i--) {
			for (int j = i + 1; j < n; j++) {
				if (s.charAt(i) == s.charAt(j)) {
					if (j > i + 2) {
						dp[i][j] = dp[i + 1][j - 1];
					}
				} else {
					dp[i][j] = Math.min(dp[i + 1][j], dp[i][j - 1]) + 1;
				}
			}
		}
		return dp[0][n - 1];
	}

	/*
	 * 1) Find the length of LCS of input string and its reverse. Let the length
	 * be ‘l’. 2) The minimum number insertions needed is length of input string
	 * minus ‘l’.
	 */

	public int findMinInsertionsLCS(String s) {
		int lcs = findLCS(s, new StringBuilder(s).reverse().toString());
		return s.length() - lcs;
	}

	public int findLCS(String s1, String s2) {
		int n = s1.length();
		int[][] dp = new int[n + 1][n + 1];

		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= n; j++) {
				if (s1.charAt(i - 1) == s2.charAt(j - 1))
					dp[i][j] = dp[i - 1][j - 1] + 1;
				else
					dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
			}
		}
		return dp[n][n];
	}

	/*
	 * 给一个unsorted slots with
	 * numbers，比如：【3,1,0,2,x，4】，x表示位置为空。每次只能移动数字到空的位置，不能直接两数字互换。
	 * 让排序这个slots，并且唯一的空在最后，比如【0,1,2,3,4，x】。数字只包含 0到n-2
	 */

	public int[] sortSlots(int[] nums) {
		int slot = -1;
		for (int i = 0; i < nums.length; i++) {
			if (nums[i] == -1)
				slot = i;
		}

		for (int i = 0; i < nums.length; i++) {
			while (nums[i] != i) {
				if (i == slot || nums[i] == nums[nums[i]])
					break;
				nums[slot] = nums[i];
				if (slot == nums[i]) {
					nums[i] = -1;
					slot = i;
				} else {
					slot = nums[i];
					nums[i] = nums[nums[i]];
					nums[slot] = -1;
				}
				System.out.println("p:" + Arrays.toString(nums) + " i=" + i
						+ " slot=" + slot);

			}
			// break;
		}
		// System.out.println(Arrays.toString(nums));
		return nums;
	}

	/*
	 * 实现一个function，输入是一个string，string里的每一个字符都是‘0-9’，所以它每个substring都可以表示一个整数。
	 * 输出其最长的可以被三整除的substring的长度。 例如“1012300‘， 最长的能被三整除的substring是
	 * ’012300‘，长度为6，所以返回值为6.
	 */

	public int longestSubstringDivisibleBy3(String num) {
		int n = num.length();
		if (n == 0)
			return 0;
		int max = 0;
		int start = 0, end = 0;
		int[] remainder = { -1, -1, -1 };
		int sum = 0;
		for (int i = 0; i < num.length(); i++) {
			sum += num.charAt(i) - '0';
			if (sum % 3 == 0) {
				max = i + 1;
				start = 0;
				end = i;
			}
			int rem = sum % 3;
			// if array, negative number is possible, should check this
			// if(rem<0)
			// rem*=-1;
			if (remainder[rem] == -1)
				remainder[rem] = i;
			else {
				if (i - remainder[rem] > max) {
					max = i - remainder[rem];
					start = remainder[rem] + 1;
					end = i;
				}
			}
		}
		System.out.println(num.substring(start, end + 1));
		return max;
	}

	/*
	 * Continuous Subsequence divisible by a number Problem Statement: What is
	 * the number of sub-sequences divisible by the given divisor ?
	 * 
	 * Given sequence:
	 * 
	 * {1 , 1 , 3 , 2 , 4, 1 ,4 , 5} Divisor is 5
	 * 
	 * Answer is 11
	 */

	public int numOfSubsequences(int[] nums, int d) {
		int res = 0;
		int[] hash = new int[d];
		int sum = 0;

		for (int i = 0; i < nums.length; i++) {
			if (nums[i] % d == 0)
				res++;
			sum += nums[i];
			int rem = sum % d;
			if (rem < 0)
				rem *= -1;
			hash[rem]++;
		}

		for (int i = 0; i < d; i++) {
			int count = hash[i];
			if (count > 1)
				res += count * (count - 1) / 2;
		}
		res += hash[0];
		return res;
	}

	/*
	 * 写一个function， 输入一个char[][], 一个int len。char[][] 里每个字符都是distinct的。输出int
	 * count。 count 的含义是 char array 里能生成的所有长度为len的code的个数。 code 的含义是可以通过char[][]
	 * 生成的字符串。code里的下一个字符可以是当前字符的上/下/左/右字符，也可以是它自己。但是不能是‘*’或者‘#’。 举个例子： input:
	 * char[][]: a b c d e f * # x len: 2 可能的code有： aa, ab, ad, bb, ba, bc, be,
	 * cc, cb, cf, dd, da, de, ee, eb, ed, ef, ff, fc, fe, fx, xx 一共22个可能的code，
	 * 输出 count = 22。
	 */
	public int numOfCodes(char[][] grid, int len) {
		int n = grid.length;
		if (n == 0)
			return 0;
		int m = grid[0].length;

		int[] count = { 0 };
		List<String> allCodes = new ArrayList<String>();
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (grid[i][j] != '*' && grid[i][j] != '#') {
					numOfCodesUtil(grid, len, i, j, count, new StringBuilder(
							grid[i][j]), allCodes);
				}
			}
		}
		System.out.println(allCodes);
		return count[0];
	}

	public void numOfCodesUtil(char[][] grid, int len, int i, int j,
			int[] count, StringBuilder code, List<String> allCodes) {
		if (i < 0 || j < 0 || i >= grid.length || j >= grid[0].length
				|| grid[i][j] == '*' || grid[i][j] == '#'
				|| code.length() > len)
			return;
		code.append(grid[i][j]);
		if (code.length() == len) {
			allCodes.add(code.toString());
			count[0]++;
		}
		numOfCodesUtil(grid, len, i, j, count, code, allCodes);
		numOfCodesUtil(grid, len, i + 1, j, count, code, allCodes);
		numOfCodesUtil(grid, len, i - 1, j, count, code, allCodes);
		numOfCodesUtil(grid, len, i, j + 1, count, code, allCodes);
		numOfCodesUtil(grid, len, i, j - 1, count, code, allCodes);
		code.deleteCharAt(code.length() - 1);
	}

	/*
	 * 判断一个树是不是另一个的subTree
	 */

	public boolean isSubTree(TreeNode root1, TreeNode root2) {
		if (root2 == null)
			return true;
		if (root1 == null)
			return false;
		if (isIdentical(root1, root2))
			return true;
		return isSubTree(root1.left, root2) || isSubTree(root1.right, root2);
	}

	public boolean isIdentical(TreeNode root1, TreeNode root2) {
		if (root1 == null && root2 == null)
			return true;
		if (root1 == null || root2 == null)
			return false;
		if (root1.val != root2.val)
			return false;
		return isIdentical(root1.left, root2.left)
				&& isIdentical(root1.right, root2.right);
	}

	/*
	 * 一个string decompression的题。 2[abc]3[a]c => abcabcabcaaac; 2[ab3[d]]2[cc] =>
	 * abdddabdddcc
	 */
	public String decompression(String s) {
		if (s.length() < 3)
			return s;
		Stack<Character> stk = new Stack<Character>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (c != ']')
				stk.push(c);
			else {
				StringBuilder sb = new StringBuilder();
				while (stk.peek() != '[') {
					sb.insert(0, stk.pop());
				}
				stk.pop();
				int count = getCount(stk);
				for (int j = 0; j < count; j++) {
					for (int k = 0; k < sb.length(); k++) {
						stk.push(sb.charAt(k));
					}
				}
			}
		}
		StringBuilder res = new StringBuilder();
		while (!stk.isEmpty()) {
			res.insert(0, stk.pop());
		}
		return res.toString();
	}

	public int getCount(Stack<Character> stk) {
		StringBuilder sb = new StringBuilder();
		while (!stk.isEmpty()) {
			if (stk.peek() >= '0' && stk.peek() <= '9') {
				sb.insert(0, stk.pop());
			} else
				break;
		}
		return Integer.valueOf(sb.toString());
	}

	// recursion
	public String decompressionRecur(String s) {
		if (!s.contains("[") || !s.contains("]"))
			return s;
		int openIndx = s.indexOf('[');
		int i = openIndx - 1;
		for (; i >= 0; i--) {
			if (s.charAt(i) < '0' || s.charAt(i) > '9')
				break;
		}
		int num = Integer.valueOf(s.substring(i + 1, openIndx));
		StringBuilder sb = new StringBuilder();
		int count = 1;
		int j = openIndx + 1;
		for (; j < s.length(); j++) {
			if (s.charAt(j) == '[')
				count++;
			else if (s.charAt(j) == ']')
				count--;
			if (count == 0) {
				break;
			}
		}
		sb.append(s.substring(0, i + 1));
		for (int k = 0; k < num; k++) {
			sb.append(s.substring(openIndx + 1, j));
		}
		sb.append(s.substring(j + 1));

		return decompressionRecur(sb.toString());
	}

	/*
	 * Next Greater Element Given an array, print the Next Greater Element (NGE)
	 * for every element. The Next greater Element for an element x is the first
	 * greater element on the right side of x in array. Elements for which no
	 * greater element exist, consider next greater element as -1.
	 */
	public void nextGreaterElement(int[] A) {
		int n = A.length;
		if (n == 0)
			return;
		Stack<Integer> stk = new Stack<Integer>();
		for (int i = 0; i < n; i++) {
			int next = A[i];
			while (!stk.isEmpty() && next > stk.peek()) {
				System.out.println(stk.pop() + "--->" + next);
			}
			stk.push(next);
		}
		while (!stk.isEmpty()) {
			System.out.println(stk.pop() + "--->" + -1);
		}
	}

	/*
	 * Binary Tree Longest Consecutive Sequence 10min就写完了。。
	 * 
	 * 然后follow up 是 问你如果依次减小也算的话， 求出最长路径。。很简单啦
	 */
	public int longestConsecutive(TreeNode root) {
		if (root == null)
			return 0;
		int inc = Math.max(longestConsecutiveUtilInc(root.left, 1, root.val),
				longestConsecutiveUtilInc(root.right, 1, root.val));
		int dec = Math.max(longestConsecutiveUtilDec(root.left, 1, root.val),
				longestConsecutiveUtilDec(root.right, 1, root.val));
		return Math.max(inc, dec);
	}

	// increasing
	public int longestConsecutiveUtilInc(TreeNode root, int count, int val) {
		if (root == null)
			return count;
		count = root.val == val + 1 ? count + 1 : 1;
		int left = longestConsecutiveUtilInc(root.left, count, root.val);
		int right = longestConsecutiveUtilInc(root.right, count, root.val);
		return Math.max(count, Math.max(left, right));
	}

	// decreasing
	public int longestConsecutiveUtilDec(TreeNode root, int count, int val) {
		if (root == null)
			return count;
		count = root.val == val - 1 ? count + 1 : 1;
		int left = longestConsecutiveUtilDec(root.left, count, root.val);
		int right = longestConsecutiveUtilDec(root.right, count, root.val);
		return Math.max(count, Math.max(left, right));
	}

	/*
	 * Consider a row of n coins of values v1 . . . vn, where n is even. We play
	 * a game against an opponent by alternating turns. In each turn, a player
	 * selects either the first or last coin from the row, removes it from the
	 * row permanently, and receives the value of the coin. Determine the
	 * maximum possible amount of money we can definitely win if we move first.
	 */
	/*
	 * F(i, j) represents the maximum value the user can collect from i'th coin
	 * to j'th coin.
	 * 
	 * F(i, j) = Max(Vi + min(F(i+2, j), F(i+1, j-1) ), Vj + min(F(i+1, j-1),
	 * F(i, j-2) )) Base Cases F(i, j) = Vi If j == i F(i, j) = max(Vi, Vj) If j
	 * == i+1
	 */
	public int optimalStrategyOfGame(int[] coins) {
		int n = coins.length;
		int[][] dp = new int[n][n];

		for (int gap = 0; gap < n; gap++) {
			for (int i = 0; i < n - gap; i++) {
				int j = i + gap;
				if (i == j)
					dp[i][j] = coins[i];
				else if (i == j - 1)
					dp[i][j] = Math.max(coins[i], coins[j]);
				else {
					dp[i][j] = Math
							.max(coins[i]
									+ Math.min(dp[i + 2][j], dp[i + 1][j - 1]),
									coins[j]
											+ Math.min(dp[i][j - 2],
													dp[i + 1][j - 1]));
				}
			}
		}
		return dp[0][n - 1];
	}

	/*
	 * 首先给一个字典，比如：{apple, people,...} 再给一个misspelling word，比如：adple，返回它的正确拼写，即
	 * apple 还知道一个限制条件，misspelling
	 * word只跟原单词有一个字母的区别。如果输入是addle，返回null。如果字数不同，也返回null
	 */
	public String findCorrectWord(Set<String> dict, String word) {
		for (String s : dict) {
			if (s.length() != word.length())
				continue;
			int count = 0;
			for (int i = 0; i < s.length(); i++) {
				if (s.charAt(i) != word.charAt(i)) {
					if (++count > 1)
						break;
				}
			}
			if (count == 1)
				return s;
		}
		return null;
	}

	public String findCorrectWord2(Set<String> dict, String word) {
		char[] wordArray = word.toCharArray();
		for (char c = 'a'; c <= 'z'; c++) {
			for (int i = 0; i < wordArray.length; i++) {
				if (wordArray[i] != c) {
					char t = wordArray[i];
					wordArray[i] = c;
					String s = new String(wordArray);
					if (dict.contains(s))
						return s;
					wordArray[i] = t;
				}
			}
		}
		return null;
	}

	/*
	 * Give you an array of integers: A Goal is to find three indexes (i,j,k)
	 * such that A[i] + A[j] == A[k]
	 * 
	 * For Example: A = [ -5, 10, 1, 8, -2 ] 10 + -2 == 8 Good answer: i=1, j=4,
	 * k=3
	 */
	public int[] findTriple(int[] nums) {
		int[] res = { -1, -1, -1 };
		if (nums.length < 3)
			return res;
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < nums.length; i++) {
			map.put(nums[i], i);
		}

		for (int i = 0; i < nums.length - 1; i++) {
			for (int j = i + 1; j < nums.length; j++) {
				int sum = nums[i] + nums[j];
				if (map.containsKey(sum) && map.get(sum) != i
						&& map.get(sum) != j) {
					res[0] = i;
					res[1] = j;
					res[2] = map.get(sum);
				}
			}
		}
		return res;
	}

	/*
	 * 第一轮: 最简单contains
	 * duplicates，注意在没有dup的时候返回值的处理，如果返回类型是int，就不能返回null。要把返回值改成Integer
	 * 加了限制条件：1.所有数字>=1,<=n-1 2.sorted 3.只有一组duplicates binary
	 * search：分隔条件是，1，2，3，4，5，5 index： 0，1，2，3，4，5 可以看到一个数字如果前面没有dup num=i+1，否则
	 * num=i,以此为比较条件
	 */

	public int findDuplicate(int[] nums) {
		int beg = 0, end = nums.length - 1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			int count = 0;
			for (int i = 0; i <= mid; i++) {
				if (nums[i] <= mid)
					count++;
			}
			if (count <= mid)
				beg = mid + 1;
			else
				end = mid - 1;
		}
		return nums[beg];
	}

	// 给一个string看能不能构成palindrome 比如level是 lveel是 空串儿是
	// a是 ab不是 题很快做出来了 followup: pring all possible palindromes

	public boolean canBePalindrome(String s) {
		if (s.length() < 2)
			return true;
		Map<Character, Integer> count = new HashMap<Character, Integer>();
		for (char c : s.toCharArray()) {
			if (!count.containsKey(c))
				count.put(c, 0);
			count.put(c, count.get(c) + 1);
		}
		int odd = 0;
		for (Map.Entry<Character, Integer> entry : count.entrySet()) {
			if (entry.getValue() % 2 == 1)
				odd++;
		}
		return odd <= 1;
	}

	public List<String> allPossiblePalindromes(String s) {
		List<String> res = new ArrayList<String>();
		Map<Character, Integer> map = new HashMap<Character, Integer>();
		for (char c : s.toCharArray()) {
			if (!map.containsKey(c))
				map.put(c, 0);
			map.put(c, map.get(c) + 1);
		}
		int odd = 0;
		StringBuilder sb = new StringBuilder();
		char c = ' ';
		for (Map.Entry<Character, Integer> entry : map.entrySet()) {
			int count = entry.getValue();
			if (count % 2 == 1) {
				odd++;
				c = entry.getKey();
			}
			for (int i = 0; i < count / 2; i++) {
				sb.append(entry.getKey());
			}
		}
		if (odd > 1)
			return res;
		List<String> half = getPermutations(sb.toString());

		for (String p : half) {
			StringBuilder stringBuilder = new StringBuilder(p)
					.append(new StringBuilder(p).reverse());
			if (odd == 1) {
				stringBuilder.insert(stringBuilder.length() / 2, c);
			}
			res.add(stringBuilder.toString());

		}

		return res;
	}

	public List<String> getPermutations(String s) {
		List<String> res = new ArrayList<String>();
		boolean[] used = new boolean[s.length()];
		getPermutationsUtil(s, "", res, used);
		return res;
	}

	public void getPermutationsUtil(String s, String sol, List<String> res,
			boolean[] used) {
		if (sol.length() == s.length()) {
			res.add(sol);
		}
		for (int i = 0; i < s.length(); i++) {
			if (!used[i]) {
				if (i != 0 && s.charAt(i) == s.charAt(i - 1) && !used[i - 1])
					continue;
				used[i] = true;
				getPermutationsUtil(s, sol + s.charAt(i), res, used);
				used[i] = false;
			}
		}
	}

	/*
	 * 一个string，如何插入数目最少的字符使得它变成panlindrome.
	 */

	public int minInsertionToPalindrome(String s) {
		if (s.length() < 2)
			return 0;
		int n = s.length();
		int[][] dp = new int[n][n];
		for (int i = n - 2; i >= 0; i--) {
			for (int j = i + 1; j < n; j++) {
				if (s.charAt(i) == s.charAt(j)) {
					if (i + 1 < j - 1)
						dp[i][j] = dp[i + 1][j - 1];
				} else {
					if (i < j - 1)
						dp[i][j] = Math.min(dp[i + 1][j], dp[i][j - 1]) + 1;
					else
						dp[i][j] = 1;
				}
			}
		}

		return dp[0][n - 1];
	}

	// not done yet
	public int minInsertionToPalindrome2(String s) {
		int n = s.length();
		int[][] dp = new int[n][n];

		for (int gap = 1; gap < n; gap++) {
			for (int i = 0; i < n - gap; i++) {
				int j = i + gap;
				if (j == i + 1) {
					if (s.charAt(i) == s.charAt(j))
						dp[i][j] = 0;
					else {
						dp[i][j] = 1;
					}
				} else {
					if (s.charAt(i) == s.charAt(j))
						dp[i][j] = dp[i + 1][j - 1];
					else
						dp[i][j] = Math.min(dp[i + 1][j], dp[i][j - 1]) + 1;
				}
			}
		}
		for (int i = 0; i < n; i++) {
			System.out.println(Arrays.toString(dp[i]));
		}

		// print the palindrome:
		int i = 0, j = n - 1;
		String p = "";
		while (i < j) {
			if (dp[i][j] == dp[i + 1][j - 1]) {
				p = p + s.charAt(i);
				i++;
				j--;
			} else if (dp[i][j] == dp[i + 1][j] + 1) {
				p = p + s.charAt(i);
				i++;
			} else if (dp[i][j] == dp[i][j - 1] + 1) {
				p = s.charAt(j) + p;
				j--;
			}
		}
		StringBuilder palindrome = new StringBuilder(p)
				.append(new StringBuilder(p).reverse());

		return dp[0][n - 1];
	}

	public List<List<String>> partition(String s) {
		List<List<String>> res = new ArrayList<List<String>>();
		List<String> sol = new ArrayList<String>();
		dfs(s, sol, res, 0);
		return res;
	}

	public void dfs(String s, List<String> sol, List<List<String>> res, int cur) {
		System.out.println(sol);
		if (cur == s.length()) {
			res.add(new ArrayList<String>(sol));
		}
		for (int i = cur; i < s.length(); i++) {
			String s1 = s.substring(cur, i + 1);
			if (isPal(s1)) {
				sol.add(s1);
				dfs(s, sol, res, i + 1);
				sol.remove(sol.size() - 1);
			}
		}
	}

	public boolean isPal(String s) {
		if (s.length() < 2)
			return true;
		int i = 0, j = s.length() - 1;
		while (i < j) {
			if (s.charAt(i) != s.charAt(j))
				return false;
			i++;
			j--;
		}
		return true;
	}

	// 有一个double类型的数组， 找满足 [a, a + 1) 的最长序列含有的元素的个数， eg. [ 1.0 ,1.3 ,1.5 ,2.3,
	// 3.5], 最长的是[1.0 1.3 1.5], 应该返回3
	public int longestSequence(double[] nums) {
		int start = 0;
		int max = 1;
		for (int i = 1; i < nums.length; i++) {
			if (nums[i] < nums[start] + 1) {
				max = Math.max(max, i - start + 1);
			} else {
				start = i;
			}
		}
		return max;
	}

	// 给一个string，只含有a和b,a可以变成b,b可以变成a,也可以不操作，返回操作次数最少就可以得到的sort的string
	public int minChange(String s) {
		int n = s.length();
		if (n < 2)
			return 0;
		int[] A = new int[n + 1];// A[i+1] counts # of 'b' for string from
									// position 0 to i
		int[] B = new int[n + 1];// B[i] counts # of 'a' for string from
									// position N-1 to i
		for (int i = 1; i <= n; i++) {
			if (s.charAt(i - 1) == 'b') {
				A[i] = A[i - 1] + 1;
			} else
				A[i] = A[i - 1];
		}

		System.out.println(Arrays.toString(A));

		for (int i = n; i > 0; i--) {
			if (s.charAt(i - 1) == 'a') {
				B[i - 1] = B[i] + 1;
			} else
				B[i - 1] = B[i];
		}
		System.out.println(Arrays.toString(B));

		int res = n;
		for (int i = 0; i < n; i++) {
			res = Math.min(res, A[i] + B[i]);
		}
		return res;
	}

	// phone第一题：.
	// input是两个string，例如："google", "algorithm"
	// output是一个string，按above例子就是："lggooe"

	public String sortStringAccordingly(String s1, String s2) {
		Map<Character, Integer> map = new HashMap<Character, Integer>();
		for (char c : s1.toCharArray()) {
			if (!map.containsKey(c))
				map.put(c, 1);
			else
				map.put(c, map.get(c) + 1);
		}
		StringBuilder sb = new StringBuilder();
		for (char c : s2.toCharArray()) {
			if (map.containsKey(c)) {
				for (int i = 0; i < map.get(c); i++) {
					sb.append(c);
				}
				map.remove(c);
			}
		}
		for (char c : s1.toCharArray()) {
			if (map.containsKey(c)) {
				sb.append(c);
			}
		}
		return sb.toString();
	}

	/*
	 * 给一个string, 找出lexical order 最小的， size==k的， subsequence, (note, not
	 * substring) String findMin(String s, k){} e.g. input s=pineapple, k==3,
	 * 
	 * output: ale ale is the lexical order smallest subsequnce of length 3.
	 */
	
	public String findMin(String s, int k){
		if(s.length()<k)
			return "";
		int count = s.length()-k;
		Stack<Character> stk=new Stack<Character>();
		for(char c: s.toCharArray()){
			while(count>0&&!stk.isEmpty()&&c<stk.peek()){
				stk.pop();
				count--;
			}
			stk.push(c);
		}
		while(count>0){
			stk.pop();
			count--;
		}
		StringBuilder sb=new StringBuilder();
		while(!stk.isEmpty()){
			sb.append(stk.pop());
		}
		return sb.reverse().toString();
	}

	/*
	 * 1.
	 * 一个房间，有个入口，有个出口，出口和入口在不同的边上。在这个房间里，有n个sensor，每个sensor有个中心，从这个中心辐射出一个圆型的探测区域
	 * 。人走到这个区域里就会发出警告，表示不能通过。要求写一个函数，check人有没有可能通过这个房间 input: 一个sensor的list,
	 * list中每个object是一个三维的list（x, y, r）x和 y是坐标，r是半径； 房间的长l和宽w。
	 */
	class Sensor {
		int x, y, r;
	}

	// public boolean canExit(List<Sensor> sensors, int l, int w, int enx, int
	// eny, int ex, int ey){
	//
	// }
	//
	/*
	 * 给一个two D garden , 每一个slot可以是flower或者Wall.
	 * 找一个合适的位置，让游客可以看到最多的flower.可以站在flower上，不能站在墙上。。
	 * 如果被墙挡了，就看不到墙后面的花。然后游客只能竖直或者水瓶看，不能看对角线。。比如 [ [f, x, x, w, f], [f, f, x ,x
	 * ,x], [x, x, f, w, f], [f, f, x, w, x]] 这样，{3, 0} 和 {1,4}都能看到四朵花。
	 */

	// public int maxFlower(char[][] garden){
	// int m=garden.length;
	// if(m==0)
	// return 0;
	//
	//
	// }

	/*
	 * there are m piles of coins, each pile is consisted of coins with
	 * different values. you are allowed to take n coins home. However, you can
	 * only take coins from the top of each pile. What's the maximum value you
	 * can obtain.
	 */

	/*
	 * 1. 游戏设计，在一个矩阵里放各种形状的水管，然后要求从走下角流入的水最后能流出右上角，如果水流入任何空的小方块（该方块没有水管，相当于漏了），
	 * 或者不能流出右上角，则失败，返回boolean， 是否能成功。想法是用一个水管就用一个map<流入口位置，流出口位置>，
	 * 每一个小方块都是一个map，然后就开始模拟水流，如果到了右上角就成功。 2. 一个array，先存有even的number， 然后存有odd
	 * 的number， 如果找到第一个odd 的number：binary search； 然后如果不知道这个array
	 * 的长度，但是如果访问boundary 外的位置，会给一个提醒（就是表明以及出界了），如何找到第一个odd number， 还是binary
	 * search， 只是一开始随便设一个Integer.MAX_VALUE, 然后两倍两倍的扩展到第一个界外元素，在开始binary search
	 * 5. longest increasing path in tree, follow up: 如何返回那个最长的path：global max 和
	 * globa path，找到长的就更新 6. moving window average 和 mixing window
	 * median：minheap和maxheap. from: 1point3acres.com/bbs
	 */

	/*
	 * 给一个float number P (e.g = 1.4523), 给定另外一个float number
	 * x，求x的sqrt是不是和P前k位相同。二分法改进一下。必要的时候用Math.ceiling()。
	 */

	/*
	 * 每一个contact信息都包含Id，Name，Phone，Email。如果两个contacts有任何一项重合就算是同一个contact。
	 * 给你一堆contacts，让你把属于同一个的分到一个group里，最后输出所有的group。这题不难，我用的Union
	 * Find。讲了一下思路后写代码
	 * ，他看了代码后应该能工作，然后讨论一下runtime。又问我有没有别的办法？我说也可以建一个graph，每个node是一个contact
	 * ，如果有任何一项重合就是一个edge。最后所有的联通的就是一个group
	 */

	/*
	 * 一个array，保存了一个图的所有edges。[[from node, to node], ], 计算bi directional
	 * edges的个数.
	 */

	/*
	 * give a list of intervals, find min number of points which will intersect
	 * all intervals
	 */

	/*
	 * design a graph class, 要求返回node的successor （children，children's children，
	 * children's children's children， etc） node的predecessor （parent，parent's
	 * parents，etc） node的direct successor （children） node的direct predecessor
	 * （parent）
	 */

	/*
	 * 2. K最常用url链接，出现无数次。 3. 两个可循环buffer相互拷贝，key point大约是解决index收尾相接。follow
	 * up是怎么地高效率，减少拷贝次数。 4. 设计一个迷宫游戏，怎么走出迷宫，BFS解决。 5.
	 * tree版本的2sum，找出一个定值，数据结构变成tree，follow up是 3sum, n sum.
	 */

	public static void main(String[] args) throws Exception {

		GoogleSolutions sol = new GoogleSolutions();
		System.out.println(sol.reArrange("aaaabbbcc", 2));

		int arr[] = { 3, 1, 5, 4, 2, 3 };
		System.out.println(sol.isSubsetSum(arr));
		System.out.println(sol.subsetSumWays(arr));

		NaryNode root = new NaryNode(1);

		NaryNode node1 = new NaryNode(2);
		NaryNode node2 = new NaryNode(3);
		NaryNode node3 = new NaryNode(4);
		root.children = Arrays.asList(node1, node2, node3);

		NaryNode node5 = new NaryNode(5);
		NaryNode node6 = new NaryNode(6);
		NaryNode node7 = new NaryNode(7);
		node2.children = Arrays.asList(node5);
		node3.children = Arrays.asList(node6, node7);
		System.out.println(sol.printTree(root));

		String s = sol.serialize(root);
		System.out.println(s);
		NaryNode r = sol.deSerialize(s);

		System.out.println(sol.printTree(r));

		System.out.println(sol.superHappyNum(72));

		int[] fees = { 3, 10, 7, 0, 4, 0, 2, 6 };
		System.out.println(sol.fewestTransactions(fees));

		int[][] mat = { { 0, 1, 0, 0 }, { 1, 1, 0, 0 }, { 0, 0, 0, 0 },
				{ 0, 0, 1, 1 }, { 0, 0, 1, 0 } };

		System.out.println(sol.findPerimeter(mat, 1, 0));

		List<Integer> nums = Arrays.asList(2, 3, 6, 9);
		System.out.println(sol.windowAverage(nums, 2));

		int[] numbers = { 2, 3, 4, 6, 9 };
		System.out.println(sol.evaluatesTo(nums, 11));
		System.out.println(sol.addOperators(numbers, 4));

		int arr1[] = { 10, 22, 9, 33, 49, 50, 31, 60 };
		System.out.println(sol.longestZigzagSequence(arr1));

		Point p1 = new Point(1, 2);
		Point p2 = new Point(3, 2);
		Point p3 = new Point(0, 5);
		Point p4 = new Point(4, 5);
		Point p5 = new Point(5, 7);
		Point p6 = new Point(-2, 7);
		Point[] ps = { p1, p2, p3, p4, p5, p6 };
		System.out.println(sol.axialSymmetric(ps));

		System.out.println(sol.minDifference(numbers));

		int[] primes = { 3, 5, 7 };
		System.out.println(Arrays.toString(sol.firstNSuperUglyNumbers(primes,
				10)));

		int A[] = { 1, 8, 7, 10, 3, 7, 12, 15 };
		System.out.println(sol.printLongestIncreasingSequence(A));
		System.out.println(sol.printAllPossibleLIS(A));

		System.out.println(sol.findExtraCharacter("123", "1243"));

		int[][] arrs = { { 2, 6, 12, 34 }, { 1, 9, 20, 1000 },
				{ 23, 34, 90, 2000 } };

		System.out.println(Arrays.toString(sol.mergeKsortedArrays(arrs, 3)));

		int[] A1 = { 0, 3 };
		int[] B1 = { 1, 0, 4 };

		sol.multiply(A1, B1);

		List<String> words = Arrays.asList("google", "leg", "apple", "about",
				"lemma", "time");
		System.out.println(sol.longestPair(words));

		System.out.println(sol
				.parseString("wo men  dou \"zai shua ti\" hao fan a"));

		System.out.println(sol.longestSubstringDivisibleBy3("11021101"));

		char[][] grid = { { 'a', 'b', 'c' }, { 'd', 'e', 'f' },
				{ '*', '#', 'x' } };

		System.out.println(sol.numOfCodes(grid, 2));

		int[] seq = { 2, 5, 1, 3, 7 };
		System.out.println(sol.numOfSubsequences(seq, 5));

		String compression = "2[ab3[d]]2[cc]";

		System.out.println(sol.decompression(compression));
		System.out.println(sol.decompressionRecur(compression) + " babababba");

		sol.nextGreaterElement(seq);

		int[] coins = { 20, 30, 2, 2, 2, 10 };
		System.out.println(sol.optimalStrategyOfGame(coins));
		Set<String> dict = new HashSet<String>(Arrays.asList("apple", "people",
				"additle", "couple"));
		System.out.println(sol.findCorrectWord(dict, "adple"));
		System.out.println(sol.findCorrectWord2(dict, "adple"));

		int[] dup = { 1, 2, 3, 4, 5, 5 };
		System.out.println(sol.findDuplicate(dup));
		System.out.println("-------------------------");
		System.out.println(sol.minInsertionToPalindrome("geeks"));
		System.out.println(sol.minInsertionToPalindrome2("geeks"));
		System.out.println(sol.minInsertionToPalindrome2("abcde"));
		int[] AA = { 5, 10, 1, 5, 8, -2 };
		System.out.println(Arrays.toString(sol.findTriple(AA)));

		System.out.println(sol.allPossiblePalindromes("ssssbba"));

		char[][] garden = { { 'f', 'x', 'x', 'w', 'f' },
				{ 'f', 'f', 'x', 'x', 'x' }, { 'x', 'x', 'f', 'w', 'f' },
				{ 'f', 'f', 'x', 'w', 'x' } };

		// System.out.println(sol.maxFlower(garden));
		System.out.println(sol.partition("ab"));
		double[] sequence = { 1.0, 1.3, 1.5, 2.3, 2.5, 2.7, 3 };
		System.out.println(sol.longestSequence(sequence));

		System.out.println(sol.minChange("bbbbaaaa"));

		System.out.println(sol.sortStringAccordingly("google", "algorithm"));
		
		System.out.println(sol.findMin("pineapple", 3));
	}

}
