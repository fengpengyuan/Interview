package com.leetcode;

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
import java.util.Scanner;
import java.util.Set;
import java.util.Stack;
import java.util.TreeMap;

public class Solution {
	static String name = "";

	public static String changeName(String s) {
		String realname = s;
		name = realname;
		return realname;
	}

	public static int longestValidParentheses(String s) {
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
        int beg=0;
        int end=nums.length-1;
        
        while(beg<=end){
            int mid=(beg+end)/2;
            if(nums[mid]==target)
                return mid;
            if(nums[mid]>target)
                end=mid-1;
            else
                beg=mid+1;
        }
        return beg;
    }

	public ArrayList<String> restoreIpAddresses(String s) {
		ArrayList<String> res = new ArrayList<String>();
		if (s.length() < 4 || s.length() > 12)
			return res;
		String sol = "";
		restoreIpAddressUtil(0, s, sol, res);
		return res;
	}

	public void restoreIpAddressUtil(int dep, String s, String sol,
			ArrayList<String> res) {
		if (dep == 3 && isValidNum(s))
			res.add(sol + s);
		for (int i = 1; i < 4 && i < s.length(); i++) {
			if (isValidNum(s.substring(0, i)))
				restoreIpAddressUtil(dep + 1, s.substring(i),
						sol + s.substring(0, i) + '.', res);
		}
	}

	public boolean isValidNum(String s) {
		if (s.charAt(0) == '0')
			return s.equals("0");
		int num = Integer.parseInt(s);
		return num > 0 && num <= 255;
	}

	public static void findTriple(int[] A) {
		if (A.length < 3)
			return;
		int n = A.length;
		int[] leftmin = new int[n];
		int[] rightmax = new int[n];
		int min = A[0];
		for (int i = 0; i < n; i++) {
			if (A[i] < min)
				min = A[i];
			leftmin[i] = min;
		}
		int max = A[n - 1];
		for (int i = n - 1; i >= 0; i--) {
			if (A[i] > max)
				max = A[i];
			rightmax[i] = max;
		}

		for (int i = 1; i < n - 1; i++) {
			if (A[i] > leftmin[i] && A[i] < rightmax[i])
				System.out.println(leftmin[i] + " " + A[i] + " " + rightmax[i]);
		}
	}

	public static int[] profitStock(int[] prices) {
		int[] res = { -1, -1 };
		if (prices.length < 2)
			return res;
		int min = prices[0];
		int max = 0;
		for (int i = 1; i < prices.length; i++) {
			if (min > prices[i])
				min = prices[i];
			if (prices[i] - min > max) {
				max = prices[i] - min;
				res[0] = min;
				res[1] = prices[i];
			}
		}
		System.out.println(Arrays.toString(res));
		return res;
	}

	public static ArrayList<String> wordBreak(String s, Set<String> dict) {
		int n = s.length();
		boolean[][] dp = new boolean[n][n + 1];

		for (int i = n - 1; i >= 0; i--) {
			for (int j = i + 1; j <= n; j++) {
				String sub = s.substring(i, j);
				if (dict.contains(sub) && j == n) {
					dp[i][j - 1] = true;
					dp[i][n] = true;
				} else {
					if (dict.contains(sub) && j < n && dp[j][n]) {
						dp[i][j - 1] = true;
						dp[i][n] = true;
					}
				}
			}
		}
		ArrayList<String> res = new ArrayList<String>();
		if (!dp[0][n])
			return res;

		wordBreakUtil(0, s, dp, "", res);
		return res;
	}

	public static void wordBreakUtil(int cur, String s, boolean[][] dp,
			String sol, ArrayList<String> res) {
		if (cur == s.length()) {
			res.add(sol);
		}

		for (int i = cur; i < s.length(); i++) {
			if (dp[cur][i]) {
				String sub = "";
				if (i < s.length() - 1)
					sub = sol + s.substring(cur, i + 1) + " ";
				else
					sub = sol + s.substring(cur, i + 1);
				wordBreakUtil(i + 1, s, dp, sub, res);
			}
		}
	}

	public static ArrayList<Integer> intersection(ArrayList<Integer> lst1,
			ArrayList<Integer> lst2) {
		ArrayList<Integer> res = new ArrayList<Integer>();
		if (lst1.size() == 0 || lst2.size() == 0)
			return res;
		for (int i = 0; i < lst1.size() && i < lst2.size(); i++) {
			int num1 = lst1.get(i);
			int num2 = lst2.get(i);
			// if (!res.contains(num1))
			// res.add(num1);
			//
			// if (!res.contains(num2))
			// res.add(num2);
			if (lst2.contains(num1) && !res.contains(num1))
				res.add(num1);
			if (lst1.contains(num2) && !res.contains(num2))
				res.add(num2);
		}
		return res;
	}

	public static void skipMdeleteN(ListNode head, int m, int n) {
		if (head == null)
			return;
		ListNode cur = head;
		while (cur != null) {
			for (int i = 0; i < m - 1 && cur != null; i++)
				cur = cur.next;
			if (cur == null)
				return;
			ListNode t = cur.next;
			for (int i = 0; i < n && t != null; i++)
				t = t.next;
			cur.next = t;
			cur = t;
		}
	}

	public static int kthLevelSum(TreeNode root, int k) {
		if (root == null)
			return 0;
		int[] sum = { 0 };
		kthLevelSum(root, 0, k, sum);
		return sum[0];
	}

	public static void kthLevelSum(TreeNode root, int cur, int k, int[] sum) {
		if (root == null)
			return;
		if (cur == k) {
			sum[0] += root.val;
		}
		kthLevelSum(root.left, cur + 1, k, sum);
		kthLevelSum(root.right, cur + 1, k, sum);
	}

	public static int getHeight(TreeNode root) {
		if (root == null)
			return 0;
		int left = getHeight(root.left);
		int right = getHeight(root.right);
		return left > right ? left + 1 : right + 1;
	}

	public static int longestPathLeafToLeaf(TreeNode root) {
		if (root == null)
			return 0;
		int left = longestPathLeafToLeaf(root.left);
		int right = longestPathLeafToLeaf(root.right);

		int both = getHeight(root.left) + getHeight(root.right) + 1;

		return Math.max(Math.max(left, right), both);

	}

	public static boolean isPalindrome(ListNode head) {
		int len = 0;
		ListNode cur = head;
		while (cur != null) {
			len++;
			cur = cur.next;
		}

		return isPalindrome(head, len);

	}

	public static boolean isPalindrome(ListNode head, int len) {
		if (head == null)
			return false;
		if (len == 1)
			return true;
		ListNode cur = head;
		for (int i = 0; i < len - 1; i++)
			cur = cur.next;

		if (head.val != cur.val)
			return false;
		return isPalindrome(head.next, len - 2);

	}

	public static boolean isPalindrome2(ListNode head) {
		// can be done via reversing the list and compare with the original one
		if (head == null)
			return false;
		if (head.next == null)
			return true;
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

		ListNode node = reverseList(second);

		ListNode cur = head;
		while (node != null) {
			if (node.val != cur.val)
				return false;
			node = node.next;
			cur = cur.next;
		}
		return true;

	}

	public static ListNode reverseList(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode pre = null;
		ListNode cur = head;
		while (cur != null) {
			ListNode next = cur.next;
			cur.next = pre;
			pre = cur;
			cur = next;
		}
		return pre;
	}

	// There is a N*N integer matrix Arr[N][N]. From the row r and column c,
	// we can go to any of the following three indices:
	// I. Arr[ r+1 ][ c-1 ] (valid only if c-1>=0)
	//
	// II. Arr[ r+1 ][ c ]
	//
	// III. Arr[ r+1 ][ c+1 ] (valid only if c+1<=N-1)
	//
	// So if we start at any column index on row 0, what is the largest sum of
	// any of the paths till row N-1.

	public static int findMaxFromFirstRow(int[][] matrix) {
		int n = matrix.length;
		if (n == 0)
			return 0;
		int m = matrix[0].length;

		int[][] dp = new int[n][m];
		for (int i = 0; i < m; i++) {
			dp[0][i] = matrix[0][i];
		}

		for (int i = 1; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (j == 0) {
					dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j + 1])
							+ matrix[i][j];
				} else if (j == m - 1) {
					dp[i][j] = Math.max(dp[i - 1][j], dp[i - 1][j - 1])
							+ matrix[i][j];
				} else
					dp[i][j] = Math.max(dp[i - 1][j],
							Math.max(dp[i - 1][j - 1], dp[i - 1][j + 1]))
							+ matrix[i][j];
			}
		}

		int max = Integer.MIN_VALUE;
		for (int i = 0; i < m; i++) {
			if (dp[n - 1][i] > max)
				max = dp[n - 1][i];
		}
		return max;
	}

	public static int[] nextGreatestElement(int[] A) {
		int n = A.length;
		int[] res = new int[n];
		res[n - 1] = -1;
		int max = A[n - 1];
		for (int i = n - 2; i >= 0; i--) {
			res[i] = max;
			if (A[i] > max)
				max = A[i];
		}
		System.out.println(Arrays.toString(res));
		return res;
	}

	public static char findFirstNonRepeating(String s) {
		if (s.length() == 0)
			return '0';
		s = s.toLowerCase();
		int[] count = new int[256];

		for (int i = 0; i < s.length(); i++) {
			count[s.charAt(i)]++;
		}

		for (int i = 0; i < s.length(); i++) {
			if (count[s.charAt(i)] == 1)
				return s.charAt(i);
		}
		return '0';
	}

	public static char findFirstNonRepeating2(String s) {
		int n = s.length();
		s = s.toLowerCase();
		CountIndex[] count = new CountIndex[256];

		for (int i = 0; i < n; i++) {
			if (count[s.charAt(i)] == null) {
				CountIndex ci = new CountIndex(1, i);
				count[s.charAt(i)] = ci;
			} else {
				count[s.charAt(i)].count++;
			}
		}

		int index = n;
		for (int i = 0; i < 256; i++) {
			if (count[i] != null) {
				if (count[i].count == 1 && index > count[i].index)
					index = count[i].index;
			}
		}
		return s.charAt(index);
	}

	public static boolean all9s(int[] num) {
		for (int i = 0; i < num.length; i++) {
			if (num[i] != 9)
				return false;
		}
		return true;
	}

	public static int[] nextSmallestPalindrome(int[] num) {
		int n = num.length;
		if (all9s(num)) {
			int[] res = new int[n + 1];
			res[0] = res[n] = 1;
			System.out.println(Arrays.toString(res));
			return res;
		}

		return generateNextSmallestPalindromeUtil(num);

	}

	public static int[] generateNextSmallestPalindromeUtil(int[] num) {
		int n = num.length;
		int mid = n / 2;
		int i = mid - 1;
		int j = n % 2 == 1 ? mid + 1 : mid;

		boolean leftsmaller = false;

		while (i >= 0 && num[i] == num[j]) {
			i--;
			j++;
		}
		if (i < 0 || num[i] < num[j])
			leftsmaller = true;

		while (i >= 0) {
			num[j++] = num[i--];
		}

		if (leftsmaller) {
			int carry = 1;
			i = mid - 1;

			if (n % 2 == 1) {
				num[mid] += carry;
				carry = num[mid] / 10;
				num[mid] %= 10;
				j = mid + 1;
			} else
				j = mid;

			while (i >= 0) {
				num[i] += carry;
				carry = num[i] / 10;
				num[i] %= 10;
				num[j++] = num[i--];
			}
		}
		System.out.println(Arrays.toString(num));
		return num;
	}

	public static int distanceBetweenNodes(TreeNode root, TreeNode node1,
			TreeNode node2) {
		if (root == null)
			return 0;
		if (node1 == node2)
			return 0;
		if (covers(root.left, node1) && covers(root.left, node2))
			return distanceBetweenNodes(root.left, node1, node2);
		else if (covers(root.right, node1) && covers(root.right, node2))
			return distanceBetweenNodes(root.right, node1, node2);
		return getLevel(root, node1) + getLevel(root, node2);
	}

	public static boolean covers(TreeNode root, TreeNode node) {
		if (root == null)
			return false;
		if (root == node)
			return true;
		return covers(root.left, node) || covers(root.right, node);
	}

	public static int getLevel(TreeNode root, TreeNode node) {
		if (root == null)
			return -1;
		return getNodeLevelUtil(root, node, 0);
	}

	public static int getNodeLevelUtil(TreeNode root, TreeNode node, int cur) {
		if (root == null)
			return -1;
		if (root == node)
			return cur;
		int downLevel = getNodeLevelUtil(root.left, node, cur + 1);
		if (downLevel != -1)
			return downLevel;
		else
			return getNodeLevelUtil(root.right, node, cur + 1);
	}

	public static void topKStrings(String[] strs, int k) {
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		for (int i = 0; i < strs.length; i++) {
			if (!map.containsKey(strs[i]))
				map.put(strs[i], 1);
			else
				map.put(strs[i], map.get(strs[i]) + 1);
		}

		Comparator<StringFreq> cp = new Comparator<StringFreq>() {

			@Override
			public int compare(StringFreq o1, StringFreq o2) {
				// TODO Auto-generated method stub
				return o1.freq - o2.freq;
			}

		};
		PriorityQueue<StringFreq> que = new PriorityQueue<StringFreq>(k, cp);
		Iterator<String> it = map.keySet().iterator();
		while (it.hasNext()) {
			String s = it.next();
			int freq = map.get(s);
			StringFreq sf = new StringFreq(s, freq);
			if (que.size() < k) {
				que.add(sf);
			} else {
				if (sf.freq > que.peek().freq) {
					que.remove();
					que.add(sf);
				}
			}
		}

		while (!que.isEmpty()) {
			System.out.print(que.remove().s + " ");
		}
	}

	public static int minMakeChange(int[] coins, int m) {
		int[] dp = new int[m + 1];
		dp[0] = 0;

		for (int i = 1; i <= m; i++) {
			int t = Integer.MAX_VALUE;
			;
			for (int j = 0; j < coins.length; j++) {
				if (coins[j] <= i) {
					t = Math.min(t, dp[i - coins[j]]);
				}
			}
			if (t < Integer.MAX_VALUE)
				dp[i] = t + 1;
			else
				dp[i] = Integer.MAX_VALUE;
		}
		return dp[m];
	}

	public static int maxProductSubarray(int[] A) {
		int max_ending_here = 1;
		int min_ending_here = 1;
		int max = 1;

		for (int i = 0; i < A.length; i++) {
			if (A[i] > 0) {
				max_ending_here *= A[i];
				min_ending_here *= A[i];
			} else if (A[i] == 0) {
				max_ending_here = 1;
				min_ending_here = 1;
			} else {
				int tmp = max_ending_here;
				max_ending_here = Math.max(1, min_ending_here * A[i]);
				min_ending_here = tmp * A[i];
			}
			max = Math.max(max_ending_here, max);
		}
		return max;
	}

	public static ListNode reverseKGroup(ListNode head, int k) {
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;

		ListNode cur = head;
		ListNode pre = dummy;

		while (cur != null) {
			for (int i = 0; i < k - 1 && cur != null; i++) {
				cur = cur.next;
			}
			if (cur == null)
				break;
			pre = reverseList(pre, cur.next);
			cur = pre.next;
		}
		return dummy.next;
	}

	public static ListNode reverseList(ListNode pre, ListNode next) {
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

	public static String serializeTree(TreeNode root) {
		if (root == null)
			return "";
		String res = "";
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		que.add(root);
		while (!que.isEmpty()) {
			TreeNode top = que.remove();
			if (top == null)
				res += "#";
			else {
				res += top.val;
				que.add(top.left);
				que.add(top.right);
			}
		}
		return res;
	}

	public static void inorder(TreeNode root) {
		if (root == null)
			return;
		inorder(root.left);
		System.out.print(root.val + " ");
		inorder(root.right);
	}

	public static TreeNode deserializeTree(String res) {
		if (res.length() == 0 || res.charAt(0) == '#')
			return null;
		int p = 0;
		TreeNode root = new TreeNode(res.charAt(p++) - '0');
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		que.add(root);
		while (!que.isEmpty()) {
			TreeNode top = que.poll();
			TreeNode left = null;
			TreeNode right = null;
			System.out.println(res.charAt(p) + " " + p);
			if (p < res.length() && res.charAt(p) != '#')
				left = new TreeNode(res.charAt(p++) - '0');
			else
				p++;
			if (p < res.length() && res.charAt(p) != '#')
				right = new TreeNode(res.charAt(p++) - '0');
			else
				p++;
			top.left = left;
			top.right = right;
			if (left != null)
				que.add(left);
			if (right != null)
				que.add(right);
		}
		return root;
	}

	// public static TreeNode getNode(String res, int[] p){
	// if(p[0]>=res.length()||res.charAt(p[0])=='#')
	// return null;
	// return new TreeNode(res.charAt(p[0])-'0');
	//
	// }

	public static TreeNode constructBST(int[] pre) {
		if (pre.length == 0)
			return null;
		return constructBSTUtil(pre, 0, pre.length - 1);
	}

	public static TreeNode constructBSTUtil(int[] pre, int beg, int end) {
		if (beg > end)
			return null;
		TreeNode root = new TreeNode(pre[beg]);
		if (beg == end)
			return root;
		int index = end + 1;
		for (int i = beg + 1; i <= end; i++) {
			if (pre[i] > root.val) {
				index = i;
				break;
			}
		}
		root.left = constructBSTUtil(pre, beg + 1, index - 1);
		root.right = constructBSTUtil(pre, index, end);
		return root;
	}

	public static void solve(char[][] board) {
		int m = board.length;
		if (m == 0)
			return;
		int n = board[0].length;
		// left
		for (int i = 0; i < m; i++) {
			if (board[i][0] == 'O')
				dfsBoard(board, i, 0);
		}
		// right

		for (int i = 0; i < m; i++) {
			if (board[i][n - 1] == 'O')
				dfsBoard(board, i, n - 1);
		}
		// top
		for (int i = 1; i < n - 1; i++) {
			if (board[0][i] == 'O')
				dfsBoard(board, 0, i);
		}
		// bottom
		for (int i = 1; i < n - 1; i++) {
			if (board[m - 1][i] == 'O')
				dfsBoard(board, m - 1, i);
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 'O')
					board[i][j] = 'X';
				if (board[i][j] == '#')
					board[i][j] = 'O';
			}
		}

		for (int i = 0; i < m; i++) {
			System.out.println(Arrays.toString(board[i]));
		}
	}

	public static void dfsBoard(char[][] board, int i, int j) {
		if (i >= 0 && i < board.length && j >= 0 && j < board[0].length
				&& board[i][j] == 'O') {
			board[i][j] = '#';
			dfsBoard(board, i + 1, j);
			dfsBoard(board, i - 1, j);
			dfsBoard(board, i, j + 1);
			dfsBoard(board, i, j - 1);
		}
	}

	public static void solve2(char[][] board) {
		int m = board.length;
		if (m == 0)
			return;
		int n = board[0].length;
		Queue<Integer> que = new LinkedList<Integer>();
		for (int i = 0; i < m; i++) {
			if (board[i][0] == 'O')
				bfsBoard(board, i, 0, que);
		}

		for (int i = 0; i < m; i++) {
			if (board[i][n - 1] == 'O')
				bfsBoard(board, i, n - 1, que);
		}

		for (int i = 1; i < n - 1; i++) {
			if (board[0][i] == 'O')
				bfsBoard(board, 0, i, que);
		}

		for (int i = 1; i < n - 1; i++) {
			if (board[m - 1][i] == 'O')
				bfsBoard(board, m - 1, i, que);
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 'O')
					board[i][j] = 'X';
				if (board[i][j] == '#')
					board[i][j] = 'O';
			}
		}
		for (int i = 0; i < m; i++) {
			System.out.println(Arrays.toString(board[i]));
		}
	}

	public static void bfsBoard(char[][] board, int i, int j, Queue<Integer> que) {
		if (i >= 0 && i < board.length && j >= 0 && j < board[0].length
				&& board[i][j] == 'O') {
			que.add(i * board.length + j);
			board[i][j] = '#';

			while (!que.isEmpty()) {
				int cur = que.poll();

				int x = cur / board.length;
				int y = cur % board.length;
				bfsBoard(board, x + 1, y, que);
				bfsBoard(board, x - 1, y, que);
				bfsBoard(board, x, y + 1, que);
				bfsBoard(board, x, y - 1, que);
			}
		}
	}

	static Queue<Integer> que = new LinkedList<Integer>();

	public static void solve3(char[][] board) {
		int m = board.length;
		if (m == 0)
			return;
		int n = board[0].length;

		for (int i = 0; i < m; i++) {
			if (board[i][0] == 'O')
				bfsBoard(board, i, 0);
		}

		for (int i = 0; i < m; i++) {
			if (board[i][n - 1] == 'O')
				bfsBoard(board, i, n - 1);
		}

		for (int i = 1; i < n - 1; i++) {
			if (board[0][i] == 'O')
				bfsBoard(board, 0, i);
		}

		for (int i = 1; i < n - 1; i++) {
			if (board[m - 1][i] == 'O')
				bfsBoard(board, m - 1, i);
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 'O')
					board[i][j] = 'X';
				if (board[i][j] == '#')
					board[i][j] = 'O';
			}
		}
		for (int i = 0; i < m; i++) {
			System.out.println(Arrays.toString(board[i]));
		}
	}

	public static void fill(char[][] board, int i, int j) {
		if (i < 0 || i >= board.length || j < 0 || j >= board[0].length
				|| board[i][j] != 'O')
			return;
		int id = i * board[0].length + j;
		que.add(id);
		board[i][j] = '#';
	}

	public static void bfsBoard(char[][] board, int i, int j) {
		fill(board, i, j);
		while (!que.isEmpty()) {
			int id = que.remove();
			int x = id / board[0].length;
			int y = id % board[0].length;
			fill(board, x + 1, y);
			fill(board, x - 1, y);
			fill(board, x, y + 1);
			fill(board, x, y - 1);
		}
	}

	public static int largestIndependentSet(TreeNode root) {
		if (root == null)
			return 0;
		int children = largestIndependentSet(root.left)
				+ largestIndependentSet(root.right);
		int parent = 1;
		if (root.left != null)
			parent += largestIndependentSet(root.left.left)
					+ largestIndependentSet(root.left.right);
		if (root.right != null)
			parent += largestIndependentSet(root.right.left)
					+ largestIndependentSet(root.right.right);
		return Math.max(parent, children);

	}

	// bst find pair sum equal give sum k
	public static boolean isPairPresent(TreeNode root, int k) {
		if (root == null)
			return false;
		Stack<TreeNode> stk1 = new Stack<TreeNode>();
		Stack<TreeNode> stk2 = new Stack<TreeNode>();
		TreeNode cur1 = root, cur2 = root;

		while (true) {
			if (cur1 != null) {
				stk1.push(cur1);
				cur1 = cur1.left;
			} else if (cur2 != null) {
				stk2.push(cur2);
				cur2 = cur2.right;
			} else if (!stk1.isEmpty() && !stk2.isEmpty()) {
				cur1 = stk1.peek();
				cur2 = stk2.peek();

				if (cur1.val >= cur2.val)
					return false;
				int sum = cur1.val + cur2.val;
				if (sum == k)
					return true;
				else if (sum < k) {
					cur1 = stk1.pop();
					cur1 = cur1.right;
					cur2 = null;
				} else {
					cur2 = stk2.pop();
					cur2 = cur2.left;
					cur1 = null;
				}
			} else
				return false;
		}
	}

	public static void getNoSiblingsNodes(TreeNode root) {
		if (root == null)
			return;
		if (root.left != null && root.right == null)
			System.out.print(root.left.val + " ");
		else if (root.left == null && root.right != null)
			System.out.print(root.right.val + " ");
		getNoSiblingsNodes(root.left);
		getNoSiblingsNodes(root.right);
	}

	public static ListNode deletNodeFromCirlularList(ListNode head, int val) {
		if (head == null)
			return null;
		if (head.next == head && head.val == val)
			return null;
		// ListNode dummy=new ListNode(0);
		// ListNode cur=head.next;
		// while(cur.next!=head){
		// cur=cur.next;
		// }
		// cur.next=dummy;
		// dummy.next=head;
		// ListNode pre=dummy;
		// cur=head;
		ListNode cur = head.next;
		ListNode pre = head;

		while (cur != head && cur.val != val) {
			pre = cur;
			cur = cur.next;
		}
		if (cur.val == val) {
			if (cur == head) {
				head = cur.next;
				pre.next = head;
			} else {
				pre.next = cur.next;
			}
		}
		return head;

	}

	// Delete node from a Doubly Circular Linked List
	public static DListNode deletNodeFromDoublyCirlularList(DListNode head,
			int val) {
		if (head == null)
			return null;
		if (head.next == head && head.val == val)
			return null;
		DListNode cur = head.next;

		while (cur != head && cur.val != val) {
			cur = cur.next;
		}

		if (cur.val == val) {
			if (cur == head) {
				head = head.next;
				head.pre = cur.pre;
				cur.pre.next = head;
			} else {
				// DListNode previous=cur.pre;
				// previous.next=cur.next;
				// cur.next.pre=previous;
				cur.pre.next = cur.next;
				cur.next.pre = cur.pre;
			}
		}
		return head;
	}

	public static void leftViewOfTree(TreeNode root) {
		if (root == null)
			return;
		int[] level = { 0 };
		leftViewOfTreeUtil(root, level, 1);
	}

	public static void leftViewOfTreeUtil(TreeNode root, int[] level, int cur) {
		if (root == null)
			return;
		if (cur > level[0]) {
			System.out.print(root.val + " ");
			level[0] = cur;
		}
		leftViewOfTreeUtil(root.left, level, cur + 1);
		leftViewOfTreeUtil(root.right, level, cur + 1);

	}

	public static int findMin3DigitNum(int N) {
		for (int i = 100; i < 1000; i++) {
			if (getProduct(i) == N)
				return i;
		}
		return -1;
	}

	public static int getProduct(int num) {
		int prod = 1;
		while (num > 0) {
			int last = num % 10;
			prod *= last;
			num /= 10;
		}
		return prod;
	}

	public static ArrayList<ArrayList<Integer>> subsetSum(int[] A, int target) {
		ArrayList<ArrayList<Integer>> res = new ArrayList<ArrayList<Integer>>();
		if (A.length == 0)
			return res;
		ArrayList<Integer> sol = new ArrayList<Integer>();
		Arrays.sort(A);
		subsetSum(0, A, sol, res, target, 0, 0);
		return res;
	}

	public static void subsetSum(int dep, int[] A, ArrayList<Integer> sol,
			ArrayList<ArrayList<Integer>> res, int target, int cur, int cursum) {
		if (dep == A.length || cursum > target)
			return;
		if (cursum == target) {
			ArrayList<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
			return;
		}

		for (int i = cur; i < A.length; i++) {
			cursum += A[i];
			sol.add(A[i]);
			subsetSum(dep + 1, A, sol, res, target, i + 1, cursum);
			cursum -= A[i];
			sol.remove(sol.size() - 1);
		}
	}

	public static TreeNodeP LCAncestor(TreeNodeP node1, TreeNodeP node2) {
		if (node1 == null || node2 == null)
			return null;
		int h1 = 0;
		TreeNodeP cur1 = node1;
		while (cur1 != null) {
			h1++;
			cur1 = cur1.parent;
		}
		int h2 = 0;
		cur1 = node2;
		while (cur1 != null) {
			h2++;
			cur1 = cur1.parent;
		}

		if (h2 > h1) {
			TreeNodeP t = node1;
			node1 = node2;
			node2 = t;
		}

		for (int i = 0; i < Math.abs(h1 - h2); i++)
			node1 = node1.parent;
		// if(h1>h2){
		// for(int i=0;i<h1-h2;i++)
		// node1=node1.parent;
		// }
		// else{
		// for(int i=0;i<h2-h1;i++)
		// node2=node2.parent;
		// }

		while (node1 != null && node2 != null) {
			if (node1 == node2)
				return node1;
			node1 = node1.parent;
			node2 = node2.parent;
		}
		return null;
	}

	public static String reverseString(String s) {
		if (s.length() < 2)
			return s;
		return reverseString(s.substring(1)) + s.charAt(0);
	}

	public static void topKLargestestNum(int[] nums, int k) {
		int n = nums.length;
		if (n < k)
			return;
		int kth = findKthLargest(nums, 0, nums.length - 1, k);

		for (int i = 0; i <= kth; i++)
			System.out.print(nums[i] + " ");
	}

	public static int findKthLargest(int[] nums, int start, int end, int k) {
		int pivot = start;
		int left = start;
		int right = end;
		while (left <= right) {
			while (left <= right && nums[left] >= nums[pivot])
				left++;
			while (left <= right && nums[right] <= nums[pivot])
				right--;
			if (left < right)
				swap(nums, left, right);
		}
		swap(nums, pivot, right);
		if (k == right + 1)
			return right;
		else if (k > right + 1)
			return findKthLargest(nums, right + 1, end, k);
		else
			return findKthLargest(nums, start, right - 1, k);

	}

	public static void swap(int[] nums, int i, int j) {
		int t = nums[i];
		nums[i] = nums[j];
		nums[j] = t;
	}

	public static TreeNodeP inorderSucc(TreeNodeP node) {
		if (node == null)
			return null;
		TreeNodeP succ = null;
		if (node.parent == null || node.right != null) {
			return leftMostChild(node.right);
		} else {

			while ((succ = node.parent) != null) {
				if (succ.left == node)
					break;
				node = succ;
			}
			return succ;
		}
	}

	public static TreeNodeP leftMostChild(TreeNodeP node) {
		if (node == null)
			return null;
		while (node.left != null)
			node = node.left;
		return node;
	}

	public static TreeNodeP inorderSuccessorBST(TreeNodeP node) {
		if (node == null)
			return null;
		if (node.right != null) {
			return leftMostChild(node.right);
		}

		TreeNodeP p = node.parent;
		while (p != null && node == p.right) {
			node = p;
			p = p.parent;
		}
		return p;
	}

	public static int[] inorderTrav(TreeNode root) {
		if (root == null)
			return null;
		int n = countNodes(root);
		int[] inorder = new int[n];
		int[] index = { 0 };
		inorderTravUtil(root, inorder, index);
		System.out.println(Arrays.toString(inorder));
		return inorder;
	}

	public static int countNodes(TreeNode root) {
		if (root == null)
			return 0;
		return countNodes(root.left) + countNodes(root.right) + 1;
	}

	public static void inorderTravUtil(TreeNode root, int[] inorder, int[] index) {
		if (root == null)
			return;
		inorderTravUtil(root.left, inorder, index);
		inorder[index[0]] = root.val;
		index[0]++;
		inorderTravUtil(root.right, inorder, index);
	}

	public static ArrayList<Pair> findPathOfMaze(int[][] maze, int startr,
			int startc) {
		ArrayList<Pair> res = new ArrayList<Pair>();
		int m = maze.length;
		if (m == 0)
			return res;
		int n = maze[0].length;
		boolean[][] used = new boolean[m][n];
		boolean t = findPathOfMaze(maze, startr, startc, used, res);
		System.out.println(t);
		return res;
	}

	public static boolean findPathOfMaze(int[][] maze, int i, int j,
			boolean[][] used, ArrayList<Pair> res) {
		if (i == maze.length - 1 && j == maze[0].length - 1)
			return true;
		if (i >= 0 && i < maze.length && j >= 0 && j < maze[0].length
				&& maze[i][j] == 0 && !used[i][j]) {
			used[i][j] = true;
			Pair p = new Pair(i, j);
			res.add(p);
			boolean t = findPathOfMaze(maze, i + 1, j, used, res)
					|| findPathOfMaze(maze, i - 1, j, used, res)
					|| findPathOfMaze(maze, i, j - 1, used, res)
					|| findPathOfMaze(maze, i, j + 1, used, res);
			if (t)
				return true;
			else {
				used[i][j] = false;
				res.remove(res.size() - 1);
			}
		}

		return false;
	}

	public static int totalNodesOfLevelK(TreeNode root, int k) {
		if (root == null)
			return 0;
		int[] count = { 0 };
		totalNodesOfLevelK(root, k, count, 0);
		return count[0];
	}

	public static void totalNodesOfLevelK(TreeNode root, int k, int[] count,
			int curlevel) {
		if (root == null)
			return;
		if (curlevel == k)
			count[0]++;
		totalNodesOfLevelK(root.left, k, count, curlevel + 1);
		totalNodesOfLevelK(root.right, k, count, curlevel + 1);
	}

	public static int findKthOfSortedMatrix(int[][] matrix, int k) {
		int m = matrix.length;
		if (m == 0)
			return -1;
		int n = matrix[0].length;
		boolean[][] visited = new boolean[m][n];
		PriorityQueue<MatrixNode> heap = new PriorityQueue<MatrixNode>(k + 1,
				new Comparator<MatrixNode>() {

					@Override
					public int compare(MatrixNode n1, MatrixNode n2) {
						// TODO Auto-generated method stub
						return n2.val - n1.val;
					}

				});
		heap.add(new MatrixNode(matrix[m - 1][n - 1], m - 1, n - 1));
		visited[m - 1][n - 1] = true;

		return findKthOfSortedMatrix(matrix, heap, visited, k, 0);
	}

	public static int findKthOfSortedMatrix(int[][] matrix,
			PriorityQueue<MatrixNode> heap, boolean[][] visited, int k,
			int count) {
		MatrixNode top = heap.poll();
		count++;
		if (count == k)
			return top.val;

		if (top.i - 1 >= 0 && !visited[top.i - 1][top.j]) {
			heap.offer(new MatrixNode(matrix[top.i - 1][top.j], top.i - 1,
					top.j));
			visited[top.i - 1][top.j] = true;
		}
		if (top.j - 1 >= 0 && !visited[top.i][top.j - 1]) {
			heap.offer(new MatrixNode(matrix[top.i][top.j - 1], top.i,
					top.j - 1));
			visited[top.i][top.j - 1] = true;
		}
		return findKthOfSortedMatrix(matrix, heap, visited, k, count);
	}

	public static int kthLargestOfSortedMatrix(int[][] matrix, int k) {
		int m = matrix.length;
		if (m == 0)
			return -1;
		int n = matrix[0].length;
		PriorityQueue<Integer> heap = new PriorityQueue<Integer>(k + 1,
				new Comparator<Integer>() {

					@Override
					public int compare(Integer o1, Integer o2) {
						// TODO Auto-generated method stub
						return o2 - o1;
					}

				});
		for (int i = n - 1; i >= 0; i--)
			heap.add(matrix[m - 1][i]);
		int num = Integer.MIN_VALUE;
		for (int i = m - 2; i >= 0; i--) {
			for (int j = n - 1; j >= 0; j--) {
				num = heap.poll();
				k--;
				if (k == 0)
					return num;
				heap.offer(matrix[i][j]);
			}
		}
		while (!heap.isEmpty() && k != 0) {
			num = heap.poll();
			k--;
		}
		if (k == 0)
			return num;
		else
			return Integer.MIN_VALUE;
	}

	public static int findPathWithMaxSum(int[][] matrix) {
		int m = matrix.length;
		if (m == 0)
			return -1;
		int n = matrix[0].length;
		int[][] dp = new int[m][n];
		ArrayList<Pair> list = new ArrayList<Pair>();
		dp[0][0] = matrix[0][0];
		for (int i = 1; i < m; i++) {
			dp[i][0] = dp[i - 1][0] + matrix[i][0];
		}

		for (int j = 1; j < n; j++)
			dp[0][j] = dp[0][j - 1] + matrix[0][j];
		HashMap<Pair, Pair> map = new HashMap<Pair, Pair>();
		Pair p = null;
		Pair tp = null;
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]) + matrix[i][j];
				p = new Pair(i, j);
				if (dp[i - 1][j] > dp[i][j - 1])
					tp = new Pair(i - 1, j);
				else
					tp = new Pair(i, j - 1);
				map.put(p, tp);
			}
		}

		int min = Integer.MAX_VALUE;

		while (map.containsKey(p)) {
			list.add(map.get(p));
			min = Math.min(min, matrix[map.get(p).first][map.get(p).second]);
			p = map.get(p);
		}
		System.out.println(list);
		return min;
	}

	public static int findMinOfRotatedArray(int[] A) {
		if (A.length == 0)
			return -1;
		return findMinOfRotatedArray(A, 0, A.length - 1);
	}

	public static int findMinOfRotatedArray(int[] A, int beg, int end) {
		if (A.length == 0)
			return -1;
		if (A[beg] < A[end])
			return beg;
		int mid = beg + (end - beg) / 2;
		if (A[beg] == A[mid])
			return A[beg] < A[end] ? beg : end;
		if (A[beg] > A[mid])
			return findMinOfRotatedArray(A, beg, mid);
		else
			return findMinOfRotatedArray(A, mid + 1, end);

	}

	public static void quickSort(int[] A) {
		if (A.length < 2)
			return;
		quickSort(A, 0, A.length - 1);
		System.out.println(Arrays.toString(A));
	}

	public static void quickSort(int[] A, int beg, int end) {
		if (beg > end)
			return;
		int pivot = partition(A, beg, end);
		quickSort(A, beg, pivot - 1);
		quickSort(A, pivot + 1, end);
	}

	public static int partition(int[] A, int beg, int end) {
		int pivot = beg;
		int left = beg;
		int right = end;

		while (left <= right) {
			while (left <= right && A[left] <= A[pivot])
				left++;
			while (left <= right && A[right] > A[pivot])
				right--;
			if (left < right) {
				swap(A, left, right);
				// left++;
				// right--;
			}
		}
		swap(A, pivot, right);
		return right;
	}

	// two sum
	public static boolean hasArrayTwoCandidates(int A[], int target) {
		if (A.length < 2)
			return false;
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();

		for (int i = 0; i < A.length; i++) {
			int t = target - A[i];
			if (map.containsKey(t))
				return true;
			else
				map.put(A[i], i);
		}
		return false;
	}

	public static ListNode reverseListRecur(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode pnext = head.next;
		head.next = null;
		ListNode node = reverseListRecur(pnext);
		pnext.next = head;
		return node;

	}

	public static ListNode reverseListIterative(ListNode head) {
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

	public static boolean anagram(String s1, String s2) {
		if (s1.length() != s2.length())
			return false;
		int[] letters = new int[256];

		for (int i = 0; i < s1.length(); i++)
			letters[s1.charAt(i)]++;

		for (int i = 0; i < s2.length(); i++) {
			char c = s2.charAt(i);
			if (letters[c] == 0)
				return false;
			letters[c]--;
		}

		for (int i = 0; i < 256; i++) {
			if (letters[i] != 0)
				return false;
		}
		return true;
	}

	public static boolean anagram2(String s1, String s2) {
		if (s1.length() != s2.length())
			return false;
		int[] letters = new int[256];
		int uniqueNum = 0;
		int completeNum = 0;

		for (int i = 0; i < s1.length(); i++) {
			if (letters[s1.charAt(i)] == 0)
				uniqueNum++;
			letters[s1.charAt(i)]++;
		}

		for (int i = 0; i < s2.length(); i++) {
			char c = s2.charAt(i);
			if (letters[c] == 0)
				return false;
			letters[c]--;
			if (letters[c] == 0) {
				completeNum++;
				if (uniqueNum == completeNum)
					return i == s2.length() - 1;
			}
		}

		// for(int i=0;i<256;i++){
		// if(letters[i]!=0)
		// return false;
		// }
		return false;
	}

	public static boolean leapfrog(int[] A) {
		if (A.length == 0)
			return true;
		boolean[] visited = new boolean[A.length];
		// visited[0]=true;

		for (int i = 0; i < A.length;) {
			if (visited[i])
				return false;
			i += A[i];
		}
		return true;
	}

	public static void mergeSort(int[] A) {
		if (A.length < 2)
			return;
		mergeSort(A, 0, A.length - 1);
		System.out.println(Arrays.toString(A));
	}

	public static void mergeSort(int[] A, int beg, int end) {
		if (beg >= end)
			return;
		int mid = (beg + end) / 2;
		mergeSort(A, beg, mid);
		mergeSort(A, mid + 1, end);

		int i = beg;
		int j = mid + 1;

		while (i <= mid && j <= end) {
			if (A[i] > A[j]) {
				int t = A[j];
				// Move the left array right one position to
				// make room for the smaller number
				for (int k = j - 1; k >= i; k--)
					A[k + 1] = A[k];
				// Put the smaller number where it belongs
				A[i] = t;
				// The right array and the middle need to be
				// shifted right
				j++;
				mid++;
			}
			// No matter what the left array moves right
			i++;
		}
	}

	public static int maxProfit(int[] prices) {
		if (prices.length < 2)
			return 0;
		int min = prices[0];
		int max = Integer.MIN_VALUE;

		for (int i = 1; i < prices.length; i++) {
			if (prices[i] - min > max)
				max = prices[i] - min;
			if (prices[i] < min)
				min = prices[i];
		}
		return max;
	}

	public int maxProfit4(int k, int[] prices) {
		int n = prices.length;
		int[][] dp = new int[n + 1][k + 1];

		for (int i = 1; i <= n; i++) {
			for (int j = 1; j <= k; j++) {
				int cmax = 0;
				for (int p = i; p >= j; p--) {
					cmax = Math.max(cmax, prices[i - 1] - prices[p - 1]);
					dp[i][j] = Math.max(dp[i][j], dp[p - 1][j - 1] + cmax);
				}
			}
		}
		int max = 0;
		for (int i = 0; i <= n; i++) {
			for (int j = 0; j <= k; j++) {
				max = Math.max(max, dp[i][j]);
			}
		}
		return max;
	}

	public boolean isBST(TreeNode root) {
		if (root == null)
			return true;
		return isBSTUtil(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
	}

	public boolean isBSTUtil(TreeNode root, int leftmost, int rightmost) {
		if (root == null)
			return true;
		if (root.val <= leftmost || root.val >= rightmost)
			return false;
		return isBSTUtil(root.left, leftmost, root.val)
				&& isBSTUtil(root.right, root.val, rightmost);

	}

	public static TreeNode LCAncestor(TreeNode root, TreeNode node1,
			TreeNode node2) {
		if (root == null)
			return null;
		if (root == node1 || root == node2)
			return root;
		TreeNode left = LCAncestor(root.left, node1, node2);
		TreeNode right = LCAncestor(root.right, node1, node2);

		if (left != null && right != null)
			return root;
		return left == null ? right : left;
	}

	public static int findDistance(TreeNode root, int n1, int n2) {
		if (root == null)
			return -1;
		int[] d1 = { -1 };
		int[] d2 = { -1 };
		int[] dist = { 0 };

		TreeNode lca = findDistanceUtil(root, n1, n2, d1, d2, dist, 1);
		if (d1[0] != -1 && d2[0] != -1)
			return dist[0];
		if (d1[0] != -1) {
			return findLevel(lca, n2, 0);
		}
		if (d2[0] != -1) {
			return findLevel(lca, n1, 0);
		}
		return -1;
	}

	public static int findLevel(TreeNode root, int n, int level) {
		if (root == null)
			return -1;
		if (root.val == n)
			return level;
		int l = findLevel(root.left, n, level + 1);
		return l == -1 ? findLevel(root.right, n, level + 1) : l;
	}

	public static TreeNode findDistanceUtil(TreeNode root, int n1, int n2,
			int[] d1, int[] d2, int[] dist, int level) {
		if (root == null)
			return null;
		if (root.val == n1) {
			d1[0] = level;
			return root;
		}
		if (root.val == n2) {
			d2[0] = level;
			return root;
		}
		TreeNode left = findDistanceUtil(root.left, n1, n2, d1, d2, dist,
				level + 1);
		TreeNode right = findDistanceUtil(root.right, n1, n2, d1, d2, dist,
				level + 1);

		if (left != null && right != null) {
			dist[0] = d1[0] + d2[0] - 2 * level;
			return root;
		}
		return left == null ? right : left;
	}

	// linkedin interview
	public static char findNextChar(char[] list, char c) {
		if (list.length == 0)
			return '0';

		int beg = 0;
		int end = list.length - 1;
		char res = list[0];
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (list[mid] == c) {
				if (mid + 1 < list.length)
					return list[mid + 1];
				else
					return res;
			} else if (list[mid] < c)
				beg = mid + 1;
			else {
				res = list[mid];
				end = mid - 1;
			}
		}
		return res;

	}

	// linkedin interview
	public static int findDistanceBetweenWords(String sentence, String s1,
			String s2) {
		if (sentence.length() < s1.length() + s2.length())
			return -1;
		if (s1.equals(s2))
			return 0;
		int minDis = Integer.MAX_VALUE;
		String[] strs = sentence.split(" ");
		int pre = -1;
		for (int i = 0; i < strs.length; i++) {
			if (strs[i].equals(s1) || strs[i].equals(s2)) {
				pre = i;
				break;
			}
		}
		if (pre == -1)
			return -1;
		boolean found = false;

		for (int i = pre + 1; i < strs.length; i++) {
			if (strs[i].equals(s1) || strs[i].equals(s2)) {
				if (!strs[i].equals(strs[pre]) && i - pre < minDis) {
					minDis = i - pre;
					pre = i;
					found = true;
				} else
					pre = i;
			}
		}
		if (!found)
			return -1;
		return minDis;
	}

	public static int integralPartOfLog(int n) {
		int res = 0;
		while (n > 0) {
			n >>= 1;
			res++;
		}
		return res;
	}

	public static int findDepth(String s) {
		if (s.length() == 0)
			return 0;
		int times = -1;
		while (true) {
			times += 1;
			String str = s.replaceAll("\\(00\\)", "0");
			if (str.equals(s))
				break;
			s = str;
		}
		System.out.println(times + " " + s);

		if (times != 0 && s.equals("0"))
			times = times - 1;
		else
			times = -1;

		return times;
	}

	public static int arraySum(int[] A) {
		return arraySum(A, A.length);
	}

	public static int arraySum(int[] A, int n) {
		if (n == 0)
			return 0;
		return A[n - 1] + arraySum(A, n - 1);
	}

	// check red-black tree balanced
	public static boolean isBalanced(TreeNode root) {
		if (root == null)
			return true;
		int[] minh = { 0 };
		int[] maxh = { 0 };
		return isBalancedUtil(root, minh, maxh);
	}

	public static boolean isBalancedUtil(TreeNode root, int[] minh, int[] maxh) {
		if (root == null) {
			minh[0] = 0;
			maxh[0] = 0;
			return true;
		}
		int[] lminh = { 0 };
		int[] lmaxh = { 0 };
		int[] rminh = { 0 };
		int[] rmaxh = { 0 };

		if (isBalancedUtil(root.left, lminh, lmaxh) == false)
			return false;
		if (isBalancedUtil(root.right, rminh, rmaxh) == false)
			return false;
		minh[0] = Math.min(lminh[0], rminh[0]);
		maxh[0] = Math.max(lmaxh[0], rmaxh[0]);

		if (maxh[0] <= minh[0] * 2)
			return true;
		return false;
	}

	public static String reverseWords(String s) {
		s = s.trim();
		if (s.length() < 2)
			return s;
		String res = "";
		char pre = '0';
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (c == ' ' && pre == ' ')
				continue;
			pre = c;
			res += c;
		}
		int i = 0;
		int j = res.length() - 1;
		char[] ch = res.toCharArray();
		reverse(ch, i, j);

		int start = 0;
		for (int k = 0; k < ch.length; k++) {
			if (ch[k] == ' ') {
				reverse(ch, start, k - 1);
				start = k + 1;
			}
		}
		reverse(ch, start, j);

		return new String(ch);

	}

	public static void reverse(char[] ch, int i, int j) {
		if (i >= j)
			return;
		while (i < j) {
			char c = ch[i];
			ch[i] = ch[j];
			ch[j] = c;
			i++;
			j--;
		}
	}

	public static String reverseWords2(String s) {
		s = s.trim();
		if (s.length() < 2)
			return s;
		String[] strs = s.split(" ");
		String res = "";
		for (int i = strs.length - 1; i >= 0; i--) {
			if (!strs[i].equals("")) {
				res += strs[i] + " ";
			}
		}

		return res.substring(0, res.length() - 1);
	}

	public static int findGroupsMultipleOf3(int[] arr) {
		if (arr.length < 2)
			return 0;
		int[] count = new int[3];

		for (int i = 0; i < arr.length; i++) {
			count[arr[i] % 3]++;
		}
		int res = 0;

		// 2 from remainder 0
		res += count[0] * (count[0] - 1) / 2;
		// 1 from remainder 1 and 1 from remainder 2

		res += count[1] * count[2];

		// 3 from remainder 0;
		res += count[0] * (count[0] - 1) * (count[0] - 2) / 6;
		// 1 from 0 and 1 from 1 and 1 from 2;

		res += count[0] * count[1] * count[2];
		// 3 from remainder 1
		res += count[1] * (count[1] - 1) * (count[1] - 2) / 6;
		// 3 from remainder 2
		res += count[2] * (count[2] - 1) * (count[2] - 2) / 6;

		return res;
	}

	public static int printkdistanceNode(TreeNode root, TreeNode target, int k) {
		if (root == null)
			return -1;
		if (root == target) {
			printkdistanceNodeDown(root, k);
			return 0;
		}

		int dl = printkdistanceNode(root.left, target, k);
		if (dl != -1) {
			if (dl + 1 == k)
				System.out.print(root.val + " ");
			else
				printkdistanceNodeDown(root.right, k - dl - 2);
			return dl + 1;
		}

		int dr = printkdistanceNode(root.right, target, k);

		if (dr != -1) {
			if (dr + 1 == k)
				System.out.print(root.val + " ");
			else
				printkdistanceNodeDown(root.left, k - dr - 2);
			return dr + 1;
		}
		return -1;
	}

	public static void printkdistanceNodeDown(TreeNode root, int k) {
		if (root == null || k < 0)
			return;
		if (k == 0)
			System.out.print(root.val + " ");
		printkdistanceNodeDown(root.left, k - 1);
		printkdistanceNodeDown(root.right, k - 1);
	}

	public static boolean match(String s1, String s2) {
		if (s1.length() == 0 && s2.length() == 0)
			return true;
		if (s1.length() > 1 && s1.charAt(0) == '*' && s2.length() == 0)
			return false;
		if ((s1.length() > 0 && s2.length() > 0)
				&& (s1.charAt(0) == '?' || s1.charAt(0) == s2.charAt(0)))
			return match(s1.substring(1), s2.substring(1));
		if (s1.length() > 0 && s1.charAt(0) == '*')
			return match(s1.substring(1), s2) || match(s1, s2.substring(1));
		return false;
	}

	public static void printRightView(TreeNode root) {
		if (root == null)
			return;
		int[] maxLevel = { 0 };
		printRightViewUtil(root, maxLevel, 1);
	}

	public static void printRightViewUtil(TreeNode root, int[] maxLevel,
			int level) {
		if (root == null)
			return;
		if (maxLevel[0] < level) {
			System.out.print(root.val + " ");
			maxLevel[0] = level;
		}

		printRightViewUtil(root.right, maxLevel, level + 1);
		printRightViewUtil(root.left, maxLevel, level + 1);

	}

	public static String lcs(String s1, String s2) {
		int m = s1.length();
		int n = s2.length();
		if (m == 0 || n == 0)
			return "";
		int[][] dp = new int[m + 1][n + 1];
		for (int i = 1; i <= m; i++) {
			for (int j = 1; j <= n; j++) {
				if (s1.charAt(i - 1) == s2.charAt(j - 1))
					dp[i][j] = dp[i - 1][j - 1] + 1;
				else
					dp[i][j] = Math.max(dp[i - 1][j], dp[i][j - 1]);
			}
		}

		int length = dp[m][n];
		System.out.println(length);

		int i = m;
		int j = n;
		String res = "";

		while (i > 0 && j > 0) {
			if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
				res = s1.charAt(i - 1) + res;
				i--;
				j--;
			} else if (dp[i - 1][j] > dp[i][j - 1])
				i--;
			else
				j--;
		}
		return res;
	}

	static TreeNode leaf = null;

	public static TreeNode deepestLeaf(TreeNode root) {
		if (root == null)
			return null;
		int[] max = { 0 };
		deepestLeaf(root, max, 0);
		return leaf;
		// if(root.left==null&&root.right==null)
		// return root;
	}

	public static void deepestLeaf(TreeNode root, int[] max, int cur) {
		if (root == null)
			return;
		if (root.left == null && root.right == null && cur > max[0]) {
			max[0] = cur;
			leaf = root;
		}

		deepestLeaf(root.left, max, cur + 1);
		deepestLeaf(root.right, max, cur + 1);

	}

	// Naive Method: Returns length of smallest subarray with sum greater than
	// x.
	// If there is no subarray with given sum, then returns n+1
	public static int smallestSubWithSum(int arr[], int x) {
		if (arr.length == 0)
			return 0;
		int res = arr.length;
		for (int i = 0; i < arr.length; i++) {
			int sum = arr[i];
			if (sum > x)
				return 1;
			for (int j = i + 1; j < arr.length; j++) {
				sum += arr[j];
				if (sum > x && j - i + 1 < res) {
					res = j - i + 1;
					break;
				}
			}
		}
		return res;
	}

	// O(n) method
	public static int smallestSubWithSum2(int[] arr, int x) {
		if (arr.length == 0)
			return 0;
		int res = arr.length;
		int sum = 0;
		int start = 0;
		for (int i = 0; i < arr.length; i++) {
			while (sum > x && start < i) {
				sum -= arr[start];
				start++;
				if (res > i - start + 1)
					res = i - start + 1;
			}
			sum += arr[i];

		}
		return res;
	}

	// Remove minimum elements from either side such that 2*min becomes more
	// than max
	// recursion
	public static int minRemovals(int arr[]) {
		return minRemovals(arr, 0, arr.length - 1);
	}

	public static int minRemovals(int[] arr, int l, int h) {
		if (l >= h)
			return 0;
		int min = getMin(arr, l, h);
		int max = getMax(arr, l, h);
		if (2 * min > max)
			return 0;
		return Math.min(minRemovals(arr, l + 1, h), minRemovals(arr, l, h - 1)) + 1;
	}

	public static int getMin(int[] arr, int l, int h) {
		int min = arr[l];
		for (int i = l + 1; i <= h; i++) {
			if (arr[i] < min)
				min = arr[i];
		}
		return min;
	}

	public static int getMax(int[] arr, int l, int h) {
		int max = arr[l];
		for (int i = l + 1; i <= h; i++) {
			if (arr[i] > max)
				max = arr[i];
		}
		return max;
	}

	// DP

	public static int minRemovals2(int[] arr) {
		int n = arr.length;
		int[][] dp = new int[n][n];

		for (int gap = 0; gap < n; gap++) {
			for (int i = 0, j = gap; j < n; i++, j++) {
				int min = getMin(arr, i, j);
				int max = getMax(arr, i, j);

				dp[i][j] = 2 * min > max ? 0 : Math.min(dp[i + 1][j],
						dp[i][j - 1]) + 1;

			}
		}
		return dp[0][n - 1];
	}

	// Create a matrix with alternating rectangles of O and X

	public static char[][] generateMatrix0X(int m, int n) {
		char[][] matrix = new char[m][n];

		int top = 0;
		int bottom = m - 1;
		int left = 0;
		int right = n - 1;
		char X = 'X';
		while (left <= right && top <= bottom) {
			for (int i = left; i <= right; i++)
				matrix[top][i] = X;
			top++;

			for (int i = top; i <= bottom; i++)
				matrix[i][right] = X;
			right--;

			if (top <= bottom) {
				for (int i = right; i >= left; i--)
					matrix[bottom][i] = X;
				bottom--;
			}
			if (right >= left) {
				for (int i = bottom; i >= top; i--)
					matrix[i][left] = X;
				left++;
			}
			X = X == 'X' ? 'O' : 'X';

		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}
		return matrix;
	}

	public static char[][] generateMatrix0X2(int m, int n) {
		char[][] matrix = new char[m][n];
		int top = 0;
		int bottom = m - 1;
		int left = 0;
		int right = n - 1;
		char c = 'X';

		while (true) {
			for (int i = left; i <= right; i++)
				matrix[top][i] = c;
			if (++top > bottom)
				break;

			for (int i = top; i <= bottom; i++)
				matrix[i][right] = c;
			if (--right < left)
				break;

			for (int i = right; i >= left; i--)
				matrix[bottom][i] = c;
			if (--bottom < top)
				break;

			for (int i = bottom; i >= top; i--)
				matrix[i][left] = c;
			if (++left > right)
				break;

			c = c == 'X' ? 'O' : 'X';
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				System.out.print(matrix[i][j] + " ");
			}
			System.out.println();
		}

		return matrix;

	}

	public static ListNode mergeSortList(ListNode head) {
		if (head == null || head.next == null)
			return head;
		// int len=0;
		ListNode slow = head;
		ListNode fast = head;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			if (fast == null)
				break;
			slow = slow.next;
		}
		ListNode secondHalf = slow.next;
		slow.next = null;
		ListNode firstHalf = head;
		// System.out.println("firsthead is "+firstHalf.val);
		// System.out.println("secondhead is "+secondHalf.val);
		//
		ListNode firstHead = mergeSortList(firstHalf);
		ListNode secondHead = mergeSortList(secondHalf);

		ListNode res = mergeTwoLists(firstHead, secondHead);
		return res;

	}

	public static ListNode mergeTwoLists(ListNode head1, ListNode head2) {
		if (head1 == null || head2 == null)
			return head1 == null ? head2 : head1;

		ListNode cur1 = head1;
		ListNode cur2 = head2;
		ListNode dummy = new ListNode(0);
		ListNode cur = dummy;

		while (cur1 != null && cur2 != null) {
			if (cur1.val < cur2.val) {
				cur.next = cur1;
				cur1 = cur1.next;
			} else {
				cur.next = cur2;
				cur2 = cur2.next;
			}
			cur = cur.next;
		}
		if (cur1 != null)
			cur.next = cur1;
		if (cur2 != null)
			cur.next = cur2;

		return dummy.next;

	}

	// Checking if any anagram of a given string is palindrome or not

	// solution: if we check if there is at most one character with odd
	// occurrences
	// in the string we can say that we can form a palindrome from any anagram.

	public static boolean checkPalindrome(String s) {
		if (s == null || s.length() < 2)
			return true;

		int[] count = new int[26];

		for (int i = 0; i < s.length(); i++) {
			count[s.charAt(i) - 'a']++;
		}

		int oddOcc = 0;
		for (int i = 0; i < 26; i++) {
			if (oddOcc > 1)
				return false;
			if (count[i] % 2 == 1)
				oddOcc++;
		}
		return true;
	}

	// public void flatten(TreeNode root) {
	// if (root == null || root.left == null && root.right == null)
	// return;
	// TreeNode node = root.right;
	// root.right = root.left;
	// TreeNode rightMost = getRightMost(root.right);
	// rightMost.right = node;
	// root.left = null;
	// flatten(root.right);
	// }
	//
	// public TreeNode getRightMost(TreeNode root) {
	// if (root == null)
	// return null;
	// if (root.right != null)
	// return getRightMost(root.right);
	// return root;
	// }

	public int evalRPN(String[] tokens) {
		Stack<Integer> stk = new Stack<Integer>();
		for (int i = 0; i < tokens.length; i++) {
			String token = tokens[i];
			if (!token.equals("+") && !token.equals("-") && !token.equals("*")
					&& !token.equals("/")) {
				int val = Integer.valueOf(token);
				stk.push(val);
			} else {
				int op1 = stk.pop();
				int op2 = stk.pop();
				if (token.equals("+"))
					stk.push(op1 + op2);
				else if (token.equals("-"))
					stk.push(op2 - op1);
				else if (token.equals("*"))
					stk.push(op2 * op1);
				else
					stk.push(op2 / op1);
			}
		}
		return stk.pop();
	}

	public ListNode sortList(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode slow = head;
		ListNode fast = head;

		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			if (fast == null)
				break;
			slow = slow.next;
		}

		ListNode h1 = head;
		ListNode h2 = slow.next;
		slow.next = null;

		h1 = sortList(h1);
		h2 = sortList(h2);

		head = mergeSortedLists(h1, h2);
		return head;
	}

	public ListNode mergeSortedLists(ListNode head1, ListNode head2) {
		if (head1 == null || head2 == null)
			return head1 == null ? head2 : head1;
		ListNode head = new ListNode(0);
		ListNode pre = head;
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
		return head.next;
	}

	public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
		if (node == null)
			return null;
		Queue<UndirectedGraphNode> que = new LinkedList<UndirectedGraphNode>();
		que.add(node);
		HashMap<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<UndirectedGraphNode, UndirectedGraphNode>();
		UndirectedGraphNode copy = new UndirectedGraphNode(node.label);
		map.put(node, copy);

		while (!que.isEmpty()) {
			UndirectedGraphNode top = que.remove();
			List<UndirectedGraphNode> neighbors = top.neighbors;
			for (int i = 0; i < neighbors.size(); i++) {
				UndirectedGraphNode neighbor = neighbors.get(i);
				if (!map.containsKey(neighbor)) {
					que.add(neighbor);
					UndirectedGraphNode nd = new UndirectedGraphNode(
							neighbor.label);
					map.put(neighbor, nd);
					map.get(top).neighbors.add(nd);
				} else
					map.get(top).neighbors.add(map.get(neighbor));
			}
		}
		return copy;
	}

	public boolean hasCycle(ListNode head) {
		if (head == null)
			return false;
		ListNode slow = head;
		ListNode fast = head;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			slow = slow.next;
			if (fast == slow)
				break;
		}
		if (fast == null || fast.next == null)
			return false;
		return true;
	}

	public ListNode insertionSortList(ListNode head) {
		if (head == null)
			return null;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode last = head;
		ListNode cur = head.next;

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

	public List<List<Integer>> levelOrder(TreeNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (root == null)
			return res;
		int curlevel = 0;
		int nextlevel = 0;
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		que.add(root);
		curlevel++;
		List<Integer> level = new ArrayList<Integer>();

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

	public boolean isValidBST(TreeNode root) {
		if (root == null)
			return true;
		return isValidBST(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
	}

	public boolean isValidBST(TreeNode root, int leftMost, int rightMost) {
		if (root == null)
			return true;
		if (root.val <= leftMost || root.val >= rightMost)
			return false;
		return isValidBST(root.left, leftMost, root.val)
				&& isValidBST(root.right, root.val, rightMost);
	}

	public boolean isInterleave(String s1, String s2, String s3) {
		int n1 = s1.length();
		int n2 = s2.length();
		int n3 = s3.length();
		if (n1 + n2 != n3)
			return false;
		boolean[][] dp = new boolean[n1 + 1][n2 + 1];
		dp[0][0] = true;
		for (int i = 1; i <= n1; i++)
			dp[i][0] = s1.charAt(i - 1) == s3.charAt(i - 1) && dp[i - 1][0];
		for (int i = 1; i <= n2; i++)
			dp[0][i] = s2.charAt(i - 1) == s3.charAt(i - 1) && dp[0][i - 1];

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

	public int atoi(String str) {
		str = str.trim();
		if (str.isEmpty())
			return 0;

		boolean neg = false;
		boolean overflow = false;
		int res = 0;
		int i = 0;
		if (str.charAt(0) == '+')
			i++;
		if (str.charAt(0) == '-') {
			i++;
			neg = true;
		}

		while (i < str.length()) {
			int val = str.charAt(i) - '0';
			if (val >= 0 && val <= 9) {
				if (res <= (Integer.MAX_VALUE - val) / 10)
					res = res * 10 + val;
				else {
					overflow = true;
					break;
				}
			} else
				break;
			i++;
		}

		if (overflow) {
			if (neg)
				return Integer.MIN_VALUE;
			else
				return Integer.MAX_VALUE;
		}
		if (neg)
			return -res;
		return res;
	}

	public List<List<Integer>> threeSum(int[] num) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (num.length < 3)
			return res;
		Arrays.sort(num);
		for (int i = 0; i < num.length - 2; i++) {
			if (i == 0 || num[i] > num[i - 1]) {
				int beg = i + 1;
				int end = num.length - 1;
				while (beg < end) {
					int sum = num[i] + num[beg] + num[end];
					if (sum == 0) {
						List<Integer> sol = new ArrayList<Integer>();
						sol.add(num[i]);
						sol.add(num[beg]);
						sol.add(num[end]);
						// if(!res.contains(sol))
						res.add(sol);
						beg++;
						end--;
						while (beg < end && num[beg] == num[beg - 1])
							beg++;
						while (end > beg && num[end] == num[end + 1])
							end--;
					} else if (sum > 0)
						end--;
					else
						beg++;
				}
			}
		}
		return res;
	}

	public int threeSumClosest(int[] num, int target) {
		int n = num.length;
		int minDiff = Integer.MAX_VALUE;
		Arrays.sort(num);
		int res = Integer.MAX_VALUE;
		for (int i = 0; i < n - 2; i++) {
			int j = i + 1;
			int k = n - 1;
			while (j < k) {
				int sum = num[i] + num[j] + num[k];
				if (sum == target)
					return sum;
				int dif = Math.abs(sum - target);
				if (dif < minDiff) {
					minDiff = dif;
					res = sum;
				}
				if (sum > target) {
					k--;
				} else
					j++;
			}
		}
		return res;
	}

	public boolean isValid(String s) {
		if (s.length() % 2 != 0)
			return false;
		Stack<Character> stk = new Stack<Character>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (c == '(' || c == '[' || c == '{')
				stk.push(c);
			else {
				if (!stk.isEmpty()
						&& (c == ')' && stk.peek() == '(' || c == ']'
								&& stk.peek() == '[' || c == '}'
								&& stk.peek() == '{'))
					stk.pop();
				else
					return false;
			}
		}
		return stk.isEmpty();
	}

	public ListNode mergeKLists(List<ListNode> lists) {
		if (lists.size() == 0)
			return null;
		while (lists.size() > 1) {
			ListNode l1 = lists.remove(0);
			ListNode l2 = lists.remove(0);
			ListNode l = mergeTwoSortedLists(l1, l2);
			lists.add(l);

		}
		return lists.get(0);
	}

	public ListNode mergeTwoSortedLists(ListNode l1, ListNode l2) {
		if (l1 == null && l2 == null)
			return null;
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

	public List<String> generateParenthesis(int n) {
		List<String> res = new ArrayList<String>();
		if (n <= 0)
			return res;
		generateParenthesis(n, 0, 0, "", res);
		return res;
	}

	public void generateParenthesis(int n, int left, int right, String sol,
			List<String> res) {
		if (n == left && left == right) {
			res.add(sol);
			return;
		}
		if (left < n) {
			generateParenthesis(n, left + 1, right, sol + "(", res);
		}
		if (right < left) {
			generateParenthesis(n, left, right + 1, sol + ")", res);
		}
	}

	public List<String> generateParenthesisIterative(int n) {
		List<String> res = new ArrayList<String>();
		res.add("");

		for (int i = 0; i < n; i++) {
			List<String> lst = new ArrayList<String>();
			for (int j = 0; j < res.size(); j++) {
				String tmp = res.get(j);
				int pos = tmp.lastIndexOf('(');
				for (int k = pos + 1; k <= tmp.length(); k++) {
					lst.add(tmp.substring(0, k) + '(' + tmp.substring(k) + ')');
				}
			}
			res = lst;
		}
		return res;
	}
	
	public List<String> generateParenthesisDP(int n) {
		List<List<String>> dp = new ArrayList<List<String>>();
		dp.add(Arrays.asList(""));
		
		for(int i=1;i<=n;i++){
			List<String> t=new ArrayList<String>();
			for(int j=0;j<i;j++){
				List<String> inside = dp.get(j);
				List<String> tail = dp.get(i-j-1);
				for(String s1: inside){
					for(String s2: tail){
						t.add("("+s1+")"+s2);
					}
				}
			}
			dp.add(t);
		}
		return dp.get(n);
	}

	public void reorderList(ListNode head) {
		if (head == null)
			return;
		ListNode slow = head;
		ListNode fast = head;

		while (fast != null && fast.next.next != null) {
			fast = fast.next.next;
			if (fast == null)
				break;
			slow = slow.next;
		}

		ListNode secondHalf = slow.next;
		slow.next = null;
		secondHalf = reverseList2(secondHalf);

		ListNode first = head;
		while (first != null && secondHalf != null) {
			ListNode node1 = first.next;
			ListNode node2 = secondHalf.next;
			first.next = secondHalf;
			secondHalf.next = node1;
			first = node1;
			secondHalf = node2;
		}
	}

	public static ListNode reverseList2(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode cur = head;
		ListNode pre = null;
		while (cur != null) {
			ListNode next = cur.next;
			cur.next = pre;
			pre = cur;
			cur = next;
		}
		return pre;
	}

	public TreeNode sortedListToBST(ListNode head) {
		if (head == null)
			return null;
		ListNode slow = head;
		ListNode fast = head;
		ListNode pre = null;
		while (fast != null && fast.next != null) {
			fast = fast.next.next;
			if (fast == null)
				break;
			pre = slow;
			slow = slow.next;
		}

		ListNode head2 = slow.next;
		TreeNode root = new TreeNode(slow.val);
		root.right = sortedListToBST(head2);
		if (pre == null)
			return root;
		else {
			pre.next = null;
			root.left = sortedListToBST(head);
		}
		return root;
	}

	public TreeNode sortedListToBST2(ListNode head) {
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
		int mid = beg + (end - beg) / 2;
		ListNode cur = head;
		for (int i = beg; i < mid; i++)
			cur = cur.next;
		TreeNode root = new TreeNode(cur.val);
		root.left = sortedListToBST(head, beg, mid - 1);
		root.right = sortedListToBST(cur.next, mid + 1, end);
		return root;
	}

	public List<List<Integer>> generate(int numRows) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (numRows == 0)
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

	public int minimumTotal(List<List<Integer>> triangle) {
		int n = triangle.size();
		int len = triangle.get(n - 1).size();
		int[][] dp = new int[n][len];
		dp[0][0] = triangle.get(0).get(0);

		for (int i = 1; i < n; i++) {
			dp[i][0] = dp[i - 1][0] + triangle.get(i).get(0);
		}

		for (int i = 1; i < n; i++) {
			for (int j = 1; j < triangle.get(i).size(); j++) {
				if (j == triangle.get(i).size() - 1)
					dp[i][j] = dp[i - 1][j - 1] + triangle.get(i).get(j);
				else
					dp[i][j] = Math.min(dp[i - 1][j - 1], dp[i - 1][j])
							+ triangle.get(i).get(j);
			}
		}

		int min = Integer.MAX_VALUE;
		for (int i = 0; i < len; i++) {
			min = Math.min(min, dp[n - 1][i]);
		}
		return min;
	}

	public boolean isPalindrome(int x) {
		if (x < 0)
			return false;
		int base = 1;
		int t = x;
		while (t >= 10) {
			base *= 10;
			t = t / 10;
		}
		while (x > 0) {
			int first = x / base;
			int last = x % 10;
			if (first != last)
				return false;
			x %= base;
			x = x / 10;
			base /= 100;
		}
		return true;
	}

	public int search(int[] A, int target) {
		if (A.length == 0)
			return -1;
		int beg = 0;
		int end = A.length - 1;
		while (beg <= end) {
			int mid = beg + (end - beg) / 2;
			if (A[mid] == target)
				return mid;
			else if (A[beg] <= A[mid]) {
				if (A[beg] <= target && target < A[mid])
					end = mid - 1;
				else
					beg = mid + 1;
			} else {
				if (A[mid] < target && target <= A[end]) {
					beg = mid + 1;
				} else
					end = mid - 1;
			}
		}
		return -1;
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
			if (top.right != null)
				stk.push(top.right);
			if (top.left != null)
				stk.push(top.left);
		}
		return res;
	}

	public int lengthOfLongestSubstring(String s) {
		if (s.length() < 2)
			return s.length();
		int max = 0;
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		int start = 0;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (map.containsKey(c)) {
				max = Math.max(max, i - start);
				int dup = map.get(c);
				for (int j = start; j <= dup; j++)
					map.remove(s.charAt(j));
				start = dup + 1;
				map.put(c, i);
			} else
				map.put(c, i);
		}
		max = Math.max(max, s.length() - start);
		return max;
	}

	public RandomListNode copyRandomList(RandomListNode head) {
		if (head == null)
			return null;
		RandomListNode cur = head;
		while (cur != null) {
			RandomListNode next = cur.next;
			RandomListNode copy = new RandomListNode(cur.label);
			cur.next = copy;
			copy.next = next;
			cur = next;
		}

		cur = head;

		while (cur != null) {
			if (cur.random != null) {
				cur.next.random = cur.random.next;
			}
			cur = cur.next.next;
		}
		RandomListNode clone = head.next;
		cur = head;
		RandomListNode cur1 = clone;
		while (cur != null) {
			cur.next = cur1.next;
			cur = cur.next;
			if (cur != null)
				cur1.next = cur.next;
			cur1 = cur1.next;
		}
		return clone;
	}

	public List<List<Integer>> pathSum(TreeNode root, int sum) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (root == null)
			return res;
		List<Integer> sol = new ArrayList<Integer>();
		pathSumUtil(root, 0, sum, sol, res);
		return res;
	}

	public void pathSumUtil(TreeNode root, int cursum, int sum,
			List<Integer> sol, List<List<Integer>> res) {
		if (root == null)
			return;
		cursum += root.val;
		sol.add(root.val);
		if (root.left == null && root.right == null && cursum == sum) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}
		pathSumUtil(root.left, cursum, sum, sol, res);
		pathSumUtil(root.right, cursum, sum, sol, res);
		cursum -= root.val;
		sol.remove(sol.size() - 1);
	}

	public boolean isPalindrome(String s) {
		if (s.isEmpty() || s.length() < 2)
			return true;
		s = s.toLowerCase();
		int beg = 0;
		int end = s.length() - 1;

		while (beg < end) {
			while (beg < end && !Character.isLetterOrDigit(s.charAt(beg)))
				beg++;
			while (beg < end && !Character.isLetterOrDigit(s.charAt(end)))
				end--;
			if (s.charAt(beg) != s.charAt(end))
				return false;
			beg++;
			end--;
		}
		return true;
	}

	public int sumNumbers(TreeNode root) {
		return sumNumbersUtil(root, 0);
	}

	public int sumNumbersUtil(TreeNode root, int sum) {
		if (root == null)
			return 0;
		sum = sum * 10 + root.val;
		if (root.left == null && root.right == null)
			return sum;
		return sumNumbersUtil(root.left, sum) + sumNumbersUtil(root.right, sum);
	}

	public int sumNumbers2(TreeNode root) {
		if (root == null)
			return 0;
		Stack<TreeNode> stk = new Stack<TreeNode>();
		Stack<Integer> sumStk = new Stack<Integer>();
		TreeNode node = root;
		int preSum = 0;
		int sum = 0;
		while (node != null || !stk.isEmpty()) {
			while (node != null) {
				stk.push(node);
				preSum = preSum * 10 + node.val;
				sumStk.push(preSum);
				node = node.left;
			}
			if (!stk.isEmpty()) {
				node = stk.pop();
				preSum = sumStk.pop();
				if (node.left == null && node.right == null)
					sum += preSum;
				node = node.right;
			}
		}
		return sum;
	}

	// public int sumNumbers2(TreeNode root) {
	// if(root==null)
	// return 0;
	//
	// Queue<TreeNode> que=new LinkedList<TreeNode>();
	// int curlevel=0;
	// int nextlevel=0;
	// que.add(root);
	// curlevel++;
	// int sum=0;
	// int lastlevel=0;
	// int res=0;
	// while(!que.isEmpty()){
	// TreeNode top=que.poll();
	// curlevel--;
	// if(top.left==null&&top.right==null)
	// res+=top.val+lastlevel;
	// else
	// sum+=top.val+lastlevel;
	// if(top.left!=null){
	// que.add(top.left);
	// nextlevel++;
	// }
	// if(top.right!=null){
	// que.add(top.right);
	// nextlevel++;
	// }
	// if(curlevel==0){
	// curlevel=nextlevel;
	// nextlevel=0;
	// if(que.isEmpty())
	// break;
	// else{
	// lastlevel=sum*10;
	// sum=0;}
	// }
	// }
	// return res;
	// }

	public int removeDuplicates(int[] A) {
		if (A.length < 3)
			return A.length;
		int count = 1;
		int j = 1;
		for (int i = 1; i < A.length; i++) {
			if (A[i] != A[i - 1]) {
				A[j++] = A[i];
				count = 1;
			} else {
				count++;
				if (count < 3) {
					A[j++] = A[i];
				}
			}
		}
		return j;
	}

	public ListNode deleteDuplicates(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode cur = head.next;
		ListNode pre = head;

		while (cur != null) {
			if (cur.val == pre.val)
				cur = cur.next;
			else {
				pre.next = cur;
				pre = pre.next;
				cur = cur.next;
			}
		}
		pre.next = cur;
		return head;
	}

	public ListNode deleteDuplicates2(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode cur = head.next;
		ListNode pre = head;
		while (cur != null) {
			while (cur != null && cur.val == pre.val)
				cur = cur.next;
			pre.next = cur;
			pre = cur;
			if (cur != null)
				cur = cur.next;
		}
		return head;
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
				top = stk.pop();
				res.add(top.val);
				pre = top;
			}
		}
		return res;
	}

	public int evalRPN2(String[] tokens) {
		Stack<Integer> stk = new Stack<Integer>();
		for (int i = 0; i < tokens.length; i++) {
			String token = tokens[i];
			if (!token.equals("+") && !token.equals("-") && !token.equals("*")
					&& !token.equals("/")) {
				int val = Integer.valueOf(token);
				stk.push(val);
			} else {
				int op1 = stk.pop();
				int op2 = stk.pop();
				if (token.equals("+"))
					stk.push(op1 + op2);
				else if (token.equals("-"))
					stk.push(op2 - op1);
				else if (token.equals("*"))
					stk.push(op2 * op1);
				else
					stk.push(op2 / op1);
			}
		}
		return stk.pop();
	}

	class IntervalComparator implements Comparator<Interval> {

		@Override
		public int compare(Interval o1, Interval o2) {
			// TODO Auto-generated method stub
			return o1.start - o2.start;
		}

	}

	public List<Interval> merge(List<Interval> intervals) {
		if (intervals.size() < 2)
			return intervals;
		Collections.sort(intervals, new IntervalComparator());
		List<Interval> res = new ArrayList<Interval>();
		res.add(intervals.get(0));
		for (int i = 1; i < intervals.size(); i++) {
			Interval interval = intervals.get(i);
			Interval preInterval = res.get(res.size() - 1);
			if (interval.start > preInterval.end)
				res.add(interval);
			else
				preInterval.end = Math.max(interval.end, preInterval.end);
		}
		return res;
	}

	public void connect(TreeLinkNode root) {
		if (root == null)
			return;
		if (root.left != null)
			root.left.next = root.right;
		if (root.right != null && root.next != null)
			root.right.next = root.next.left;
		connect(root.left);
		connect(root.right);
	}

	public void connect2(TreeLinkNode root) {
		if (root == null)
			return;
		Queue<TreeLinkNode> que = new LinkedList<TreeLinkNode>();
		int curlevel = 0;
		int nextlevel = 0;
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
				top.next = null;
				curlevel = nextlevel;
				nextlevel = 0;
			} else {
				top.next = que.peek();
			}
		}
	}

	public int numTrees(int n) {
		if (n <= 1)
			return 1;
		int total = 0;
		for (int i = 1; i <= n; i++) {
			total += numTrees(i - 1) * numTrees(n - i);
		}
		return total;
	}

	public void sortColors(int[] A) {
		if (A.length < 2)
			return;
		int i = 0;
		int j = A.length - 1;
		int k = A.length - 1;
		while (i <= j) {
			if (A[i] == 0)
				i++;
			else if (A[i] == 1) {
				A[i] = A[j];
				A[j--] = 1;
			} else {
				A[i] = A[k];
				A[k--] = 2;
				if (j > k)
					j = k;
			}
		}
	}

	public void setZeroes(int[][] matrix) {
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

	public static String simplifyPath(String path) {
		if (path.isEmpty())
			return "/";
		String[] strs = path.split("/");

		Stack<String> stk = new Stack<String>();
		String res = "";
		for (int i = 0; i < strs.length; i++) {
			String s = strs[i];
			if (s.equals(".") || s.equals(""))
				continue;
			if (s.equals("..")) {
				if (!stk.isEmpty())
					stk.pop();
			} else {
				stk.push(s);
			}
		}

		if (stk.isEmpty())
			return "/";
		while (!stk.isEmpty()) {
			res = "/" + stk.pop() + res;
		}
		return res;
	}

	// 1 从i开始，j是当前station的指针，sum += gas[j] – cost[j]
	// （从j站加了油，再算上从i开始走到j剩的油，走到j+1站还能剩下多少油）
	// 2 如果sum < 0，说明从i开始是不行的。那能不能从i..j中间的某个位置开始呢？假设能从k (i <=k<=j)走，那么i..j <
	// 0，若k..j >=0，说明i..k – 1更是<0，那从k处就早该断开了，根本轮不到j。
	// 3 所以一旦sum<0，i就赋成j + 1，sum归零。
	// 4 最后total表示能不能走一圈。
	public int canCompleteCircuit(int[] gas, int[] cost) {
		int total = 0;
		int sum = 0;
		int index = 0;
		for (int i = 0; i < gas.length; i++) {
			sum += gas[i] - cost[i];
			if (sum < 0) {
				index = i + 1;
				sum = 0;
			}
			total += gas[i] - cost[i];
		}
		return total >= 0 ? index : -1;
	}

	public List<Integer> getRow(int rowIndex) {
		List<Integer> res = new ArrayList<Integer>();
		int[][] dp = new int[rowIndex + 1][rowIndex + 1];
		dp[0][0] = 1;
		for (int i = 1; i <= rowIndex; i++)
			dp[i][0] = 1;
		for (int i = 1; i <= rowIndex; i++) {
			for (int j = 1; j <= rowIndex; j++) {
				if (j == i) {
					dp[i][j] = 1;
				} else
					dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1];
			}
		}

		for (int i = 0; i <= rowIndex; i++) {
			res.add(dp[rowIndex][i]);
		}
		return res;
	}

	public List<Integer> getRowOkSpace(int rowIndex) {
		List<Integer> res = new ArrayList<Integer>();
		int[] dp = new int[rowIndex + 1];
		dp[0] = 1;
		for (int i = 1; i <= rowIndex; i++) {
			for (int j = i; j >= 1; j--) {
				if (j == i)
					dp[j] = 1;
				else
					dp[j] = dp[j] + dp[j - 1];
			}
		}
		for (int i = 0; i <= rowIndex; i++)
			res.add(dp[i]);
		return res;
	}

	public int uniquePathsWithObstacles(int[][] obstacleGrid) {
		int m = obstacleGrid.length;
		int n = obstacleGrid[0].length;
		int[][] dp = new int[m][n];
		dp[0][0] = obstacleGrid[0][0] == 1 ? 0 : 1;

		for (int i = 1; i < m; i++)
			dp[i][0] = obstacleGrid[i][0] == 1 ? 0 : dp[i - 1][0];
		for (int i = 1; i < n; i++)
			dp[0][i] = obstacleGrid[0][i] == 1 ? 0 : dp[0][i - 1];
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				dp[i][j] = obstacleGrid[i][j] == 1 ? 0 : dp[i - 1][j]
						+ dp[i][j - 1];
			}
		}
		return dp[m - 1][n - 1];
	}

	public int uniquePathsWithObstacles2(int[][] obstacleGrid) {
		int m = obstacleGrid.length;
		int n = obstacleGrid[0].length;

		int[] dp = new int[n];
		for (int i = 0; i < n && obstacleGrid[0][i] != 1; i++) {
			dp[i] = 1;
		}
		if (dp[0] == 0)
			return 0;
		for (int i = 1; i < m; i++) {
			if (obstacleGrid[i][0] == 1)
				dp[0] = 0;
			for (int j = 1; j < n; j++) {
				dp[j] = obstacleGrid[i][j] == 1 ? 0 : dp[j] + dp[j - 1];
			}
		}
		return dp[n - 1];
	}

	public int uniquePaths(int m, int n) {
		int[] dp = new int[n];

		for (int i = 0; i < n; i++)
			dp[i] = 1;
		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				dp[i] = dp[i] + dp[i - 1];
			}
		}
		return dp[n - 1];
	}

	public int lengthOfLastWord(String s) {
		s = s.trim();
		int i = s.length() - 1;
		int len = 0;
		while (i >= 0 && s.charAt(i) != ' ') {
			len++;
			i--;
		}
		return len;
	}

	public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
		if (intervals.size() == 0) {
			intervals.add(newInterval);
			return intervals;
		}
		List<Interval> res = new ArrayList<Interval>();
		boolean inserted = false;
		for (int i = 0; i < intervals.size(); i++) {
			Interval interval = intervals.get(i);
			if (interval.start < newInterval.start) {
				insertInterval(interval, res);
			} else {
				insertInterval(newInterval, res);
				inserted = true;
				insertInterval(interval, res);
			}
		}
		if (!inserted)
			insertInterval(newInterval, res);
		return res;
	}

	public void insertInterval(Interval interval, List<Interval> intervals) {
		if (intervals.size() == 0) {
			intervals.add(interval);
			return;
		}
		Interval interv = intervals.get(intervals.size() - 1);
		if (interv.end < interval.start)
			intervals.add(interval);
		else
			interv.end = Math.max(interv.end, interval.end);
	}

	public boolean exist(char[][] board, String word) {
		int n = board.length;
		if (n == 0)
			return false;
		int m = board[0].length;
		boolean[][] used = new boolean[n][m];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < m; j++) {
				if (board[i][j] == word.charAt(0)) {
					if (dfs(board, word, i, j, used, 0))
						return true;
				}
			}
		}
		return false;
	}

	public boolean dfs(char[][] board, String word, int i, int j,
			boolean[][] used, int cur) {
		if (cur == word.length())
			return true;
		if (i >= 0 && i < board.length && j < board[0].length && j >= 0
				&& !used[i][j] && board[i][j] == word.charAt(cur)) {
			used[i][j] = true;
			boolean res = dfs(board, word, i + 1, j, used, cur + 1)
					|| dfs(board, word, i - 1, j, used, cur + 1)
					|| dfs(board, word, i, j + 1, used, cur + 1)
					|| dfs(board, word, i, j - 1, used, cur + 1);
			if (res)
				return true;
			else
				used[i][j] = false;
		}
		return false;
	}

	public int longestValidParentheses2(String s) {
		if (s.length() < 2)
			return 0;
		Stack<Integer> stk = new Stack<Integer>();
		int max = 0;
		int last = -1;
		for (int i = 0; i < s.length(); i++) {
			if (s.charAt(i) == '(') {
				stk.push(i);
			} else {
				if (!stk.isEmpty()) {
					stk.pop();
					if (stk.isEmpty()) {
						max = Math.max(max, i - last);
					} else
						max = Math.max(max, i - stk.peek());
				} else
					last = i;
			}
		}
		return max;
	}

	public int longestValidParenthesesDP(String s) {
		int n = s.length();
		if (n < 2)
			return 0;
		int[] dp = new int[n];
		dp[n - 1] = 0;
		int max = 0;
		for (int i = n - 2; i >= 0; i--) {
			if (s.charAt(i) == '(') {
				int j = i + 1 + dp[i + 1];
				if (j < n && s.charAt(j) == ')') {
					dp[i] = dp[i + 1] + 2;
					if (j + 1 < n)
						dp[i] = dp[i] + dp[j + 1];
				}
				max = Math.max(max, dp[i]);
			}
		}
		return max;
	}

	public String countAndSay(int n) {
		if (n == 1)
			return "1";
		String res = "1";

		for (int i = 1; i < n; i++) {
			char c = res.charAt(0);
			int count = 1;
			String s = "";
			for (int j = 1; j < res.length(); j++) {
				if (res.charAt(j) == c) {
					count++;
				} else {
					s = s + count + c;
					c = res.charAt(j);
					count = 1;
				}
			}
			res = s + count + c;
		}
		return res;
	}
	
	

	public List<String> anagrams(String[] strs) {
		List<String> res = new ArrayList<String>();
		if (strs.length < 2)
			return res;
		HashMap<String, List<String>> map = new HashMap<String, List<String>>();
		for (int i = 0; i < strs.length; i++) {
			char[] ch = strs[i].toCharArray();
			Arrays.sort(ch);
			String s = new String(ch);
			if (!map.containsKey(s)) {
				List<String> anagrams = new ArrayList<String>();
				anagrams.add(strs[i]);
				map.put(s, anagrams);
			} else {
				map.get(s).add(strs[i]);
			}

		}
		Iterator<String> it = map.keySet().iterator();
		while (it.hasNext()) {
			String s = it.next();
			if (map.get(s).size() > 1)
				res.addAll(map.get(s));
		}
		return res;
	}

	public List<List<Integer>> combinationSum(int[] candidates, int target) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (candidates.length == 0)
			return res;
		List<Integer> sol = new ArrayList<Integer>();
		Arrays.sort(candidates);
		combinationSumUtil(candidates, target, res, sol, 0, 0);
		return res;
	}

	public void combinationSumUtil(int[] candidates, int target,
			List<List<Integer>> res, List<Integer> sol, int cursum, int dep) {
		if (dep == candidates.length || cursum > target)
			return;
		if (cursum == target) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}
		for (int i = dep; i < candidates.length; i++) {
			cursum += candidates[i];
			sol.add(candidates[i]);
			combinationSumUtil(candidates, target, res, sol, cursum, i);
			cursum -= candidates[i];
			sol.remove(sol.size() - 1);
		}
	}

	public List<List<Integer>> combinationSum2(int[] num, int target) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (num.length == 0)
			return res;
		List<Integer> sol = new ArrayList<Integer>();
		boolean[] used = new boolean[num.length];
		Arrays.sort(num);
		combinationSum2Util(num, target, res, sol, used, 0, 0);
		return res;
	}

	public void combinationSum2Util(int[] num, int target,
			List<List<Integer>> res, List<Integer> sol, boolean[] used,
			int cursum, int dep) {
		if (dep >= num.length || cursum > target)
			return;
		if (target == cursum) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}
		for (int i = dep; i < num.length; i++) {
			if (!used[i]) {
				if (i != 0 && num[i] == num[i - 1] && !used[i - 1])
					continue;
				cursum += num[i];
				used[i] = true;
				sol.add(num[i]);
				combinationSum2Util(num, target, res, sol, used, cursum, i);
				cursum -= num[i];
				sol.remove(sol.size() - 1);
				used[i] = false;
			}
		}
	}

	public int trap(int[] A) {
		int n = A.length;
		if (n < 2)
			return 0;
		int[] left = new int[n];
		int[] right = new int[n];
		int leftmost = A[0];

		for (int i = 0; i < A.length; i++) {
			left[i] = leftmost;
			if (A[i] > leftmost)
				leftmost = A[i];
		}

		int rightmost = A[n - 1];
		for (int i = n - 1; i >= 0; i--) {
			right[i] = rightmost;
			if (A[i] > rightmost)
				rightmost = A[i];
		}

		int total = 0;
		for (int i = 0; i < n; i++) {
			if (Math.min(left[i], right[i]) > A[i])
				total += Math.min(left[i], right[i]) - A[i];
		}
		return total;
	}

	public List<List<Integer>> subsets(int[] S) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (S.length == 0)
			return res;
		List<Integer> sol = new ArrayList<Integer>();
		Arrays.sort(S);
		subsetUtil(0, S.length, S, res, sol, 0);
		return res;
	}

	public void subsetUtil(int dep, int maxDep, int[] S,
			List<List<Integer>> res, List<Integer> sol, int curpos) {
		res.add(sol);
		if (dep == maxDep) {
			return;
		}

		for (int i = curpos; i < S.length; i++) {
			List<Integer> out = new ArrayList<Integer>(sol);
			out.add(S[i]);
			subsetUtil(dep + 1, maxDep, S, res, out, i + 1);
		}
	}

	public List<List<Integer>> subsetsWithDup(int[] num) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (num.length == 0)
			return res;
		List<Integer> sol = new ArrayList<Integer>();
		Arrays.sort(num);
		boolean[] used = new boolean[num.length];
		subsetWithDupUtil(0, num.length, num, res, sol, used, 0);
		return res;
	}

	public void subsetWithDupUtil(int dep, int maxDep, int[] num,
			List<List<Integer>> res, List<Integer> sol, boolean[] used,
			int curpos) {
		res.add(sol);
		if (dep == maxDep)
			return;
		for (int i = curpos; i < num.length; i++) {
			if (!used[i]) {
				if (i != 0 && num[i] == num[i - 1] && !used[i - 1])
					continue;
				List<Integer> out = new ArrayList<Integer>(sol);
				out.add(num[i]);
				used[i] = true;
				subsetWithDupUtil(dep + 1, maxDep, num, res, out, used, i + 1);
				used[i] = false;
			}
		}
	}
	
	public List<List<Integer>> subsetsWithDup2(int[] num) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        List<Integer> empty = new ArrayList<Integer>();
        result.add(empty);
        Arrays.sort(num);

        for (int i = 0; i < num.length; i++) {
            int dupCount = 0;
            while( ((i+1) < num.length) && num[i+1] == num[i]) {
                dupCount++;
                i++;
            }
            int prevNum = result.size();
            for (int j = 0; j < prevNum; j++) {
                List<Integer> element = new ArrayList<Integer>(result.get(j));
                for (int t = 0; t <= dupCount; t++) {
                    element.add(num[i]);
                    result.add(new ArrayList<Integer>(element));
                }
            }
        }
        return result;
    }

	public static ListNode deleteDuplicatesAll(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;

		ListNode pre = dummy;
		ListNode cur = head;
		while (cur != null) {
			boolean dup = false;
			while (cur.next != null && cur.val == cur.next.val) {
				dup = true;
				cur = cur.next;
			}
			if (dup) {
				pre.next = cur.next;
				cur = cur.next;
			} else {
				pre.next = cur;
				pre = cur;
				cur = cur.next;
			}
		}
		return dummy.next;
	}

	public List<TreeNode> generateTrees(int n) {
		List<TreeNode> res = new ArrayList<TreeNode>();
		generateTreesUtil(1, n, res);
		return res;
	}

	public void generateTreesUtil(int beg, int end, List<TreeNode> res) {
		if (beg > end) {
			res.add(null);
		}
		for (int i = beg; i <= end; i++) {
			List<TreeNode> left = new ArrayList<TreeNode>();
			generateTreesUtil(beg, i - 1, left);
			List<TreeNode> right = new ArrayList<TreeNode>();
			generateTreesUtil(i + 1, end, right);

			for (int j = 0; j < left.size(); j++) {
				for (int k = 0; k < right.size(); k++) {
					TreeNode root = new TreeNode(i);
					root.left = left.get(j);
					root.right = right.get(k);
					res.add(root);
				}
			}
		}
	}

	public String minWindow(String S, String T) {
		if (S.length() < T.length())
			return "";
		int[] needFind = new int[256];
		for (int i = 0; i < T.length(); i++) {
			needFind[T.charAt(i)]++;
		}

		int[] hasFound = new int[256];
		int count = T.length();
		int minLength = S.length() + 1;
		int windowStart = 0;
		int windowEnd = 0;
		int start = 0;
		for (int i = 0; i < S.length(); i++) {
			char c = S.charAt(i);
			if (needFind[c] == 0)
				continue;
			hasFound[c]++;
			if (hasFound[c] <= needFind[c])
				count--;
			if (count == 0) {
				while (hasFound[S.charAt(start)] > needFind[S.charAt(start)]
						|| needFind[S.charAt(start)] == 0) {
					if (hasFound[S.charAt(start)] > needFind[S.charAt(start)])
						hasFound[S.charAt(start)]--;
					start++;
				}
				if (i - start + 1 < minLength) {
					windowStart = start;
					windowEnd = i;
					minLength = i - start + 1;
				}
			}
		}
		if (count == 0)
			return S.substring(windowStart, windowEnd + 1);
		return "";

	}

	public List<List<Integer>> combine(int n, int k) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (n < k)
			return res;
		List<Integer> sol = new ArrayList<Integer>();
		combineUtil(0, k, n, sol, res, 1);
		return res;
	}

	public void combineUtil(int dep, int maxDep, int n, List<Integer> sol,
			List<List<Integer>> res, int curpos) {
		if (dep == maxDep) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}
		for (int i = curpos; i <= n; i++) {
			sol.add(i);
			combineUtil(dep + 1, maxDep, n, sol, res, i + 1);
			sol.remove(sol.size() - 1);
		}
	}

	public int longestConsecutive(int[] num) {
		Set<Integer> set = new HashSet<Integer>();
		for (int i = 0; i < num.length; i++)
			set.add(num[i]);
		int max = 1;
		for (int i = 0; i < num.length; i++) {
			max = Math.max(
					max,
					getCount(set, num[i] - 1, false)
							+ getCount(set, num[i], true));
		}
		return max;
	}

	public int getCount(Set<Integer> set, int num, boolean asec) {
		int count = 0;
		while (set.contains(num)) {
			set.remove(num);
			count++;
			if (asec)
				num++;
			else
				num--;
		}
		return count;
	}

	public void flatten(TreeNode root) {
		if (root == null || root.left == null && root.right == null)
			return;
		if (root.left != null) {
			TreeNode node = root.right;
			root.right = root.left;
			TreeNode rightMost = getRightMost(root.right);
			rightMost.right = node;
			root.left = null;
			flatten(root.right);
		} else
			flatten(root.right);
	}

	public TreeNode getRightMost(TreeNode root) {
		if (root == null)
			return null;
		if (root.right != null)
			return getRightMost(root.right);
		return root;
	}

	public void flatten2(TreeNode root) {
		if (root == null || root.left == null && root.right == null)
			return;
		if (root.left != null) {
			TreeNode right = root.right;
			root.right = root.left;
			TreeNode node = root.left;
			while (node.right != null)
				node = node.right;
			root.left = null;
			node.right = right;
		}
		flatten(root.right);
	}

	public int maxPathSum(TreeNode root) {
		if (root == null)
			return 0;
		int[] res = new int[1];
		res[0] = Integer.MIN_VALUE;
		maxPathSumUtil(root, res);
		return res[0];
	}

	public int maxPathSumUtil(TreeNode root, int[] res) {
		if (root == null)
			return 0;
		int left = maxPathSumUtil(root.left, res);
		int right = maxPathSumUtil(root.right, res);
		int single = Math.max(root.val, Math.max(left, right) + root.val);
		int arch = left + root.val + right;
		res[0] = Math.max(res[0], Math.max(single, arch));
		return single;
	}

	public int numDistinct(String S, String T) {
		int m = S.length();
		int n = T.length();
		int[][] dp = new int[m + 1][n + 1];
		for (int i = 0; i <= m; i++) {
			dp[i][0] = 1;
		}

		for (int i = 1; i <= m; i++) {
			for (int j = 1; j <= n; j++) {
				if (S.charAt(i - 1) == T.charAt(j - 1))
					dp[i][j] = dp[i - 1][j - 1] + dp[i - 1][j];
				else
					dp[i][j] = dp[i - 1][j];
			}
		}
		return dp[m][n];
	}

	public int numDistinctSpaceOP(String S, String T) {
		int[] matches = new int[S.length() * T.length()];
		matches[0] = 1;

		for (int i = 1; i <= S.length(); i++) {
			for (int j = T.length(); j > 0; i--) {
				if (S.charAt(i - 1) == T.charAt(j - 1))
					matches[j] += matches[j - 1];
				else
					matches[j] = matches[j - 1];
			}
		}
		return matches[T.length()];
	}

	public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (root == null)
			return res;
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		int curlevel = 0;
		int nextlevel = 0;
		boolean leftToRight = true;
		que.add(root);
		curlevel++;
		List<Integer> level = new ArrayList<Integer>();
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
				if (!leftToRight)
					Collections.reverse(level);
				res.add(level);
				level = new ArrayList<Integer>();
				leftToRight = !leftToRight;
			}
		}
		return res;
	}

	public List<List<Integer>> zigzagLevelOrder2(TreeNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (root == null)
			return res;

		boolean ltr = true;
		int h = getHeight(root);
		for (int i = 1; i <= h; i++) {
			List<Integer> level = new ArrayList<Integer>();
			printLevel(root, i, level, ltr);
			res.add(new ArrayList<Integer>(level));
			ltr = !ltr;
		}
		return res;
	}

	public void printLevel(TreeNode root, int level, List<Integer> sol,
			boolean ltr) {
		if (root == null)
			return;
		if (level == 1) {
			sol.add(root.val);
		} else {
			if (ltr) {
				printLevel(root.left, level - 1, sol, ltr);
				printLevel(root.right, level - 1, sol, ltr);
			} else {
				printLevel(root.right, level - 1, sol, ltr);
				printLevel(root.left, level - 1, sol, ltr);
			}
		}

	}

	public List<List<Integer>> levelOrderBottom(TreeNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (root == null)
			return res;
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		int curlevel = 0;
		int nextlevel = 0;
		que.add(root);
		curlevel++;
		List<Integer> level = new ArrayList<Integer>();
		while (!que.isEmpty()) {
			TreeNode top = que.remove();
			curlevel--;
			level.add(top.val);
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
		Collections.reverse(res);
		return res;
	}

	public int singleNumber(int[] A) {
		int res = 0;
		for (int i = 0; i < 32; i++) {
			int x = 1 << i;
			int sum = 0;
			for (int j = 0; j < A.length; j++) {
				if ((A[j] & x) != 0)
					sum++;
			}
			if (sum % 3 != 0)
				res |= x;
		}
		return res;
	}

	public int[] searchRange(int[] A, int target) {
		int[] res = { -1, -1 };
		int beg = 0;
		int end = A.length - 1;
		int index = -1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A[mid] == target) {
				index = mid;
				break;
			} else if (A[mid] < target)
				beg = mid + 1;
			else
				end = mid - 1;
		}
		if (index == -1)
			return res;
		beg = 0;
		end = index;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A[mid] == target)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		res[0] = beg;

		beg = index;
		end = A.length - 1;

		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A[mid] == target)
				beg = mid + 1;
			else
				end = mid - 1;
		}
		res[1] = beg - 1;
		return res;
	}

	public String strStr(String haystack, String needle) {
		if (haystack.length() < needle.length())
			return null;
		int len = needle.length();

		for (int i = 0; i <= haystack.length() - len; i++) {
			if (haystack.substring(i, i + len).equals(needle)) {
				return haystack.substring(i);
			}
		}
		return null;
	}

	public String strStr2(String haystack, String needle) {
		if (haystack.length() < needle.length())
			return null;
		int len = needle.length();

		for (int i = 0; i <= haystack.length() - len; i++) {
			int j = 0;
			for (; j < needle.length(); j++) {
				if (haystack.charAt(i + j) != needle.charAt(j))
					break;
			}
			if (j == needle.length())
				return haystack.substring(i);
		}
		return null;
	}

	public ListNode swapPairs(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode cur = head.next;
		ListNode pre = head;
		ListNode ppre = dummy;
		while (cur != null) {
			ListNode pnext = cur.next;
			cur.next = pre;
			ppre.next = cur;
			pre.next = pnext;

			cur = pnext;
			if (pnext != null) {
				ppre = pre;
				pre = cur;
				cur = pnext.next;
			}
		}
		return dummy.next;
	}

	public List<Integer> spiralOrder(int[][] matrix) {
		List<Integer> res = new ArrayList<Integer>();
		if (matrix.length == 0)
			return res;
		int top = 0;
		int bottom = matrix.length - 1;
		int left = 0;
		int right = matrix[0].length - 1;

		while (true) {
			for (int i = left; i <= right; i++)
				res.add(matrix[top][i]);
			if (++top > bottom)
				break;
			for (int i = top; i <= bottom; i++)
				res.add(matrix[i][right]);
			if (--right < left)
				break;
			for (int i = right; i >= left; i--)
				res.add(matrix[bottom][i]);
			if (--bottom < top)
				break;
			for (int i = bottom; i >= top; i--)
				res.add(matrix[i][left]);
			if (++left > right)
				break;
		}
		return res;
	}

	public int[][] generateMatrix(int n) {
		int[][] matrix = new int[n][n];
		int up = 0;
		int bottom = n - 1;
		int left = 0;
		int right = n - 1;
		int val = 1;
		while (true) {
			for (int i = left; i <= right; i++)
				matrix[up][i] = val++;
			if (++up > bottom)
				break;
			for (int i = up; i <= bottom; i++)
				matrix[i][right] = val++;
			if (--right < left)
				break;
			for (int i = right; i >= left; i--)
				matrix[bottom][i] = val++;
			if (--bottom < up)
				break;
			for (int i = bottom; i >= up; i--)
				matrix[i][left] = val++;
			if (++left > right)
				break;
		}
		return matrix;
	}

	public ListNode rotateRight(ListNode head, int n) {
		if (head == null)
			return null;
		int len = 0;
		ListNode cur = head;
		while (cur != null) {
			len++;
			cur = cur.next;
		}
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		n = n % len;
		if (n == 0)
			return head;
		cur = head;
		ListNode pre = dummy;
		for (int i = 0; i < len - n; i++) {
			pre = cur;
			cur = cur.next;
		}
		ListNode node = cur;
		pre.next = null;
		while (cur != null && cur.next != null)
			cur = cur.next;
		cur.next = dummy.next;
		dummy.next = node;

		return dummy.next;

	}

	public ListNode reverseBetween(ListNode head, int m, int n) {
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode ppre = dummy;
		ListNode cur = head;
		for (int i = 0; i < m - 1; i++) {
			ppre = cur;
			cur = cur.next;
		}
		ListNode pre = cur;
		ListNode start = cur;
		cur = cur.next;
		for (int i = 0; i < n - m; i++) {
			ListNode pnext = cur.next;
			cur.next = pre;
			pre = cur;
			cur = pnext;
		}

		ppre.next = pre;
		start.next = cur;

		return dummy.next;
	}

	public static int maxPoints(Point[] points) {
		if (points.length < 3)
			return points.length;
		int max = 0;
		for (int i = 0; i < points.length; i++) {
			HashMap<Double, Integer> map = new HashMap<Double, Integer>();
			int dup = 1;
			int vertical = 0;
			for (int j = i + 1; j < points.length; j++) {
				if (points[i].x == points[j].x) {
					if (points[i].y == points[j].y)
						dup++;
					else
						vertical++;
				} else {
					double k = points[i].y == points[j].y ? 0.0 : 1.0
							* (points[i].y - points[j].y)
							/ (points[i].x - points[j].x);
					if (!map.containsKey(k))
						map.put(k, 1);
					else
						map.put(k, map.get(k) + 1);
				}
			}
			// System.out.println("map size is " + map.size());
			Iterator<Double> it = map.keySet().iterator();
			while (it.hasNext()) {
				double k = it.next();
				// System.out.println(k + " " + map.get(k));
				max = Math.max(max, map.get(k) + dup);
			}
			max = Math.max(max, dup + vertical);
		}
		return max;
	}

	public int sqrt(int x) {
		if (x == 0)
			return 0;
		double last = 0;
		double res = 1;
		while (res != last) {
			last = res;
			res = (res + x / res) / 2;
		}
		return (int) res;
	}

	public static double sqrt(double x) {
		if (x == 0)
			return 0;
		double last = 0;
		double res = 1;
		while (last != res) {
			last = res;
			res = (res + x / res) / 2;
		}
		return res;
	}

	public int sqrtBinary(int x) {
		long i = 0;
		long j = x / 2 + 1;
		while (i <= j) {
			long mid = (i + j) / 2;
			if (mid * mid == x)
				return (int) mid;
			if (mid * mid < x)
				i = mid + 1;
			else
				j = mid - 1;
		}
		return (int) j;
	}

	public int sqrtBinary2(int x) {
		if (x <= 1)
			return x;
		int i = 0;
		int j = x;

		while (j - i > 1) {
			int mid = (i + j) / 2;
			if (mid == x / mid)
				return mid;
			else if (mid < x / mid)
				i = mid;
			else
				j = mid;
		}
		return i;
	}

	public boolean wordBreak2(String s, Set<String> dict) {
		boolean[] wb = new boolean[s.length() + 1];
		for (int i = 1; i <= s.length(); i++) {
			if (!wb[i] && dict.contains(s.substring(0, i))) {
				wb[i] = true;
			}
			if (wb[i]) {
				if (i == s.length()) {
					return true;
				}
				for (int j = i + 1; j <= s.length(); j++) {
					if (!wb[j] && dict.contains(s.substring(i, j))) {
						wb[j] = true;
					}
					if (j == s.length() && wb[j]) {
						return true;
					}
				}
			}
		}
		return false;
	}

	public List<List<Integer>> permute(int[] num) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		boolean[] used = new boolean[num.length];
		permute(0, num, sol, res, used);
		return res;
	}

	public void permute(int dep, int[] num, List<Integer> sol,
			List<List<Integer>> res, boolean[] used) {
		if (dep == num.length) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}

		for (int i = 0; i < num.length; i++) {
			if (!used[i]) {
				used[i] = true;
				sol.add(num[i]);
				permute(dep + 1, num, sol, res, used);
				sol.remove(sol.size() - 1);
				used[i] = false;
			}

		}
	}

	public List<List<Integer>> permuteUnique(int[] num) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		boolean[] used = new boolean[num.length];
		Arrays.sort(num);
		permuteUniqueUtil(0, num, sol, res, used);
		return res;
	}

	public void permuteUniqueUtil(int dep, int[] num, List<Integer> sol,
			List<List<Integer>> res, boolean[] used) {
		if (dep == num.length) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}

		for (int i = 0; i < num.length; i++) {
			if (!used[i]) {
				if (i != 0 && num[i] == num[i - 1] && !used[i - 1])
					continue;
				used[i] = true;
				sol.add(num[i]);
				permuteUniqueUtil(dep + 1, num, sol, res, used);
				sol.remove(sol.size() - 1);
				used[i] = false;
			}
		}
	}

	public static String getPermutation(int n, int k) {
		int[] num = new int[n];
		for (int i = 0; i < n; i++) {
			num[i] = i + 1;
		}
		for (int i = 1; i < k; i++) {
			nextPermutation(num);
		}
		String res = "";
		for (int i = 0; i < num.length; i++)
			res += num[i];

		return res;
	}

	public static void nextPermutation(int[] num) {
		int index = -1;

		for (int i = 0; i < num.length - 1; i++) {
			if (num[i] < num[i + 1])
				index = i;
		}
		if (index == -1) {
			Arrays.sort(num);
			return;
		}
		int idx = index + 1;
		for (int i = index + 1; i < num.length; i++) {
			if (num[i] > num[index])
				idx = i;
		}

		int t = num[index];
		num[index] = num[idx];
		num[idx] = t;
		int beg = index + 1;
		int end = num.length - 1;

		while (beg < end) {
			int tmp = num[beg];
			num[beg] = num[end];
			num[end] = tmp;
			beg++;
			end--;
		}
	}

	public String getPermutationWithMath(int n, int k) {
		ArrayList<Integer> num = new ArrayList<Integer>();
		int factorial = 1;
		for (int i = 1; i <= n; i++) {
			num.add(i);
			factorial *= i;
		}
		k--;

		String res = "";
		for (int i = 0; i < n; i++) {
			factorial = factorial / (n - i);
			int curIdx = k / factorial;
			k = k % factorial;

			res += num.get(curIdx);
			num.remove(curIdx);
		}
		return res;
	}

	public List<Integer> findSubstring(String S, String[] L) {
		List<Integer> res = new ArrayList<Integer>();
		int n = L.length;
		int len = L[0].length();
		if (S.length() < n * len)
			return res;
		HashMap<String, Integer> map = new HashMap<String, Integer>();
		for (String s : L) {
			if (!map.containsKey(s))
				map.put(s, 1);
			else
				map.put(s, map.get(s) + 1);
		}
		
		for (int i = 0; i <= S.length() - n * len; i++) {
			HashMap<String, Integer> found = new HashMap<String, Integer>();
			int j = 0;
			for (; j < n; j++) {
				String s = S.substring(i + j * len, i + j * len + len);
				if (!map.containsKey(s))
					break;
				if (found.containsKey(s))
					found.put(s, found.get(s) + 1);
				else
					found.put(s, 1);
				if (found.get(s) > map.get(s))
					break;
			}
			if (j == n)
				res.add(i);
		}
		return res;
	}

	TreeNode first, second, pre;

	public void recoverTree(TreeNode root) {
		first = null;
		second = null;
		pre = null;
		recoverTreeUtil(root);
		int t = first.val;
		first.val = second.val;
		second.val = t;
	}

	public void recoverTreeUtil(TreeNode root) {
		if (root == null)
			return;
		recoverTreeUtil(root.left);
		if (pre == null)
			pre = root;
		else {
			if (pre.val > root.val) {
				if (first == null)
					first = pre;
				second = root;
			}
			pre = root;
		}
		recoverTreeUtil(root.right);
	}

	public String intToRoman(int num) {
		if (num <= 0 || num > 3999)
			return "";
		String[] roman = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X",
				"IX", "V", "IV", "I" };
		int[] values = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
		String res = "";
		for (int i = 0; i < roman.length; i++) {
			while (num >= values[i]) {
				res += roman[i];
				num -= values[i];
			}
		}
		return res;
	}

	public int romanToInt(String s) {
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		map.put('I', 1);
		map.put('V', 5);
		map.put('X', 10);
		map.put('L', 50);
		map.put('C', 100);
		map.put('D', 500);
		map.put('M', 1000);
		int res = 0;
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			res += sign(s, i, map) * map.get(c);
		}
		return res;
	}

	public int sign(String s, int i, HashMap<Character, Integer> map) {
		if (i == s.length() - 1)
			return 1;
		if (map.get(s.charAt(i)) < map.get(s.charAt(i + 1)))
			return -1;
		else
			return 1;
	}

	public String multiply(String num1, String num2) {
		int l1 = num1.length();
		int l2 = num2.length();
		int[] res = new int[l1 + l2];

		for (int i = l1 - 1; i >= 0; i--) {
			int carry = 0;
			int dig1 = num1.charAt(i) - '0';
			for (int j = l2 - 1; j >= 0; j--) {
				int dig2 = num2.charAt(j) - '0';
				int prod = dig1 * dig2 + carry + res[i + j + 1];
				carry = prod / 10;
				prod = prod % 10;
				res[i + j + 1] = prod;
			}
			res[i] = carry;
		}

		String multiplication = "";

		int i = 0;
		while (i < l1 + l2 - 1 && res[i] == 0)
			i++;
		while (i < l1 + l2)
			multiplication += res[i++];
		return multiplication;

	}

	public List<String> letterCombinations(String digits) {
		List<String> res = new ArrayList<String>();
		String sol = "";
		letterCombinations(0, digits, sol, res);
		return res;
	}

	public void letterCombinations(int dep, String digits, String sol,
			List<String> res) {
		if (dep == digits.length()) {
			res.add(sol);
			return;
		}
		String s = getString(digits.charAt(dep) - '0');
		for (int i = 0; i < s.length(); i++) {
			letterCombinations(dep + 1, digits, sol + s.charAt(i), res);
		}

	}

	public String getString(int i) {
		String[] strs = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs",
				"tuv", "wxyz" };
		return strs[i];
	}

	public static List<String> letterCombinationsIterative(String digits) {
		List<String> res = new ArrayList<String>();
		String[] strs = { "", "", "abc", "def", "ghi", "jkl", "mno", "pqrs",
				"tuv", "wxyz" };
		res.add("");

		for (int i = 0; i < digits.length(); i++) {
			int num = digits.charAt(i) - '0';
			List<String> lst = new ArrayList<String>();
			for (int j = 0; j < res.size(); j++) {
				String tmp = res.get(j);
				for (int k = 0; k < strs[num].length(); k++) {
					lst.add(tmp + strs[num].charAt(k));
				}
			}
			res = lst;
		}
		return res;
	}

	public int jump(int[] nums) {
        if(nums.length<2)
            return 0;
        int step=1;
        int max=nums[0];
        int min=0;
        while(max<nums.length-1){
            int t=max;
            for(int i=min;i<=t;i++){
                if(i+nums[i]>max){
                    max=i+nums[i];
                    min=i;
                }
            }
            step++;
        }
        return step;
    }

	public String longestPalindrome(String s) {
		if (s.length() < 2)
			return s;
		int beg = 0;
		int end = 0;
		int maxLen = 0;
		for (int i = 1; i < s.length(); i++) {
			int j = i - 1;
			int k = i;

			while (j >= 0 && k < s.length() && s.charAt(j) == s.charAt(k)) {
				j--;
				k++;
			}
			if (k - j + 1 > maxLen) {
				beg = j + 1;
				end = k - 1;
				maxLen = k - j + 1;
			}

			j = i - 1;
			k = i + 1;
			while (j >= 0 && k < s.length() && s.charAt(j) == s.charAt(k)) {
				j--;
				k++;
			}
			if (k - j + 1 > maxLen) {
				beg = j + 1;
				end = k - 1;
				maxLen = k - j + 1;
			}
		}
		return s.substring(beg, end + 1);
	}

	public int firstMissingPositive(int[] A) {
		for (int i = 0; i < A.length; i++) {
			while (A[i] != i + 1) {
				if (A[i] <= 0 || A[i] > A.length || A[i] == A[A[i] - 1])
					break;
				int t = A[i];
				A[i] = A[A[i] - 1];
				A[t - 1] = t;
			}
		}

		for (int i = 0; i < A.length; i++) {
			if (A[i] != i + 1)
				return i + 1;
		}
		return A.length + 1;
	}

	public int firstMissingPositive2(int[] A) {
		for (int i = 0; i < A.length; i++) {
			if (A[i] > 0 && A[i] <= A.length) {
				if (A[i] != i + 1 && A[i] != A[A[i] - 1]) {
					int t = A[i];
					A[i] = A[A[i] - 1];
					A[t - 1] = t;
					i--;
				}
			}
		}

		for (int i = 0; i < A.length; i++) {
			if (A[i] != i + 1)
				return i + 1;
		}
		return A.length + 1;
	}

	public String longestPalindrome2(String s) {
		if (s.length() < 2)
			return s;
		String longest = "";
		for (int i = 0; i < s.length(); i++) {
			String str = palindromeUtil(s, i - 1, i);
			if (str.length() > longest.length())
				longest = str;

			str = palindromeUtil(s, i - 1, i + 1);
			if (str.length() > longest.length())
				longest = str;
		}
		return longest;
	}

	public String palindromeUtil(String s, int beg, int end) {
		while (beg >= 0 && end < s.length() && s.charAt(beg) == s.charAt(end)) {
			beg--;
			end++;
		}
		return s.substring(beg + 1, end);
	}

	public int maximalRectangle(char[][] matrix) {
		int m = matrix.length;
		if (m == 0)
			return 0;
		int n = matrix[0].length;

		int[][] dp = new int[m][n];

		for (int i = 0; i < m; i++) {
			dp[i][0] = matrix[i][0] - '0';
		}

		for (int i = 0; i < m; i++) {
			for (int j = 1; j < n; j++) {
				if (matrix[i][j] == '1')
					dp[i][j] = dp[i][j - 1] + 1;
				else
					dp[i][j] = 0;
			}
		}
		int maxArea = 0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				int maxLen = dp[i][j];
				for (int k = i; k >= 0; k--) {
					if (dp[k][j] == 0)
						break;
					maxLen = Math.min(maxLen, dp[k][j]);
					maxArea = Math.max(maxArea, maxLen * (i - k + 1));
				}
			}
		}
		return maxArea;
	}

	public int maximalRectangle2(char[][] matrix) {
		int m = matrix.length;
		if (m == 0)
			return 0;
		int n = matrix[0].length;

		int[][] height = new int[m][n + 1];
		for (int i = 0; i < n; i++) {
			if (matrix[0][i] == '0')
				height[0][i] = 0;
			else
				height[0][i] = 1;
		}

		for (int i = 1; i < m; i++) {
			for (int j = 0; j < n; j++) {
				height[i][j] = matrix[i][j] == '0' ? 0 : height[i - 1][j] + 1;
			}
		}
		int max = 0;
		for (int i = 0; i < m; i++) {
			int area = maxAreaInHist(height[i]);
			max = Math.max(max, area);
		}
		return max;
	}

	public int maxAreaInHist(int[] height) {
		int max = 0;
		Stack<Integer> stk = new Stack<Integer>();
		int i = 0;
		while (i < height.length) {
			if (stk.isEmpty() || height[i] >= height[stk.peek()])
				stk.push(i++);
			else {
				int top = stk.pop();
				int len = stk.isEmpty() ? i : i - stk.peek() - 1;
				max = Math.max(max, len * height[top]);
			}
		}
		return max;
	}

	public int numDecodings(String s) {
		int[] num = { 0 };
		numDecodingsUtil(s, num);
		return num[0];
	}

	public void numDecodingsUtil(String s, int[] num) {
		if (s.length() == 0)
			num[0]++;
		for (int i = 0; i <= 1 && i < s.length(); i++) {
			if (isValidNumber(s.substring(0, i + 1)))
				numDecodingsUtil(s.substring(i + 1), num);
		}
	}

	public boolean isValidNumber(String s) {
		if (s.charAt(0) == '0')
			return false;
		int num = Integer.parseInt(s);
		return num >= 1 && num <= 26;
	}

	public int numDecodingsDP(String s) {
		if (s.length() == 0)
			return 0;
		int n = s.length();
		int[] dp = new int[n + 1];
		dp[0] = 1;
		if (isValidNumber(s.substring(0, 1)))
			dp[1] = 1;

		for (int i = 2; i <= s.length(); i++) {
			if (isValidNumber(s.substring(i - 1, i)))
				dp[i] = dp[i - 1];
			if (isValidNum(s.substring(i - 2, i)))
				dp[i] += dp[i - 2];
		}
		return dp[n];
	}

	public boolean isValidSudoku(char[][] board) {
		for (int i = 0; i < 9; i++) {
			boolean[] row = new boolean[10];
			for (int j = 0; j < 9; j++) {
				if (board[i][j] == '.')
					continue;
				int num = board[i][j] - '0';
				if (num > 0 && num <= 9) {
					if (row[num])
						return false;
					row[num] = true;
				}
			}
		}

		for (int i = 0; i < 9; i++) {
			boolean[] col = new boolean[10];
			for (int j = 0; j < 9; j++) {
				if (board[j][i] == '.')
					continue;
				int num = board[j][i] - '0';
				if (col[num])
					return false;
				col[num] = true;
			}
		}

		for (int i = 0; i < 9; i += 3) {
			for (int j = 0; j < 9; j += 3) {
				boolean[] grid = new boolean[10];
				for (int k = 0; k < 3; k++) {
					for (int l = 0; l < 3; l++) {
						if (board[i + k][j + l] == '.')
							continue;
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

	public List<String> restoreIpAddresses2(String s) {
		List<String> res = new ArrayList<String>();
		if (s.length() < 4 || s.length() > 12)
			return res;
		restoreIpAddressesUtil(0, s, "", res);
		return res;
	}

	public void restoreIpAddressesUtil(int dep, String s, String sol,
			List<String> res) {
		if (dep == 3 && isValidNumber2(s)) {
			res.add(sol + s);
			return;
		}

		for (int i = 1; i < 4 && i < s.length(); i++) {
			String sub = s.substring(0, i);
			if (isValidNumber2(sub)) {
				restoreIpAddressesUtil(dep + 1, s.substring(i),
						sol + sub + ".", res);
			}
		}
	}

	public boolean isValidNumber2(String s) {
		if (s.charAt(0) == '0')
			return s.equals("0");
		int num = Integer.parseInt(s);
		return num >= 1 && num <= 255;
	}

	public List<Integer> grayCode(int n) {
		List<Integer> res = new ArrayList<Integer>();
		if (n < 0)
			return res;
		if (n == 0) {
			res.add(0);
			return res;
		}
		List<Integer> partial = grayCode(n - 1);
		res.addAll(partial);
		for (int i = partial.size() - 1; i >= 0; i--) {
			res.add(res.get(i) + (int) (Math.pow(2, n - 1)));
		}
		return res;
	}

	public String convert(String s, int nRows) {
		if (nRows < 2)
			return s;
		String res = "";
		int zigSize = 2 * nRows - 2;
		for (int i = 0; i < nRows; i++) {
			for (int j = i; j < s.length(); j += zigSize) {
				res += s.charAt(j);
				if (i > 0 && i < nRows - 1 && j + zigSize - 2 * i < s.length())
					res += s.charAt(j + zigSize - 2 * i);
			}
		}
		return res;
	}

	List<String> topSort(String[] tickets) {
		HashMap<String, String> m = new HashMap<>();
		String start = null;
		Comparator<String> c = String.CASE_INSENSITIVE_ORDER;
		for (String s : tickets) {
			String[] items = s.split("-");
			if (start == null)
				start = items[0];
			else
				start = c.compare(items[0], start) < 0 ? items[0] : start;
			m.put(items[0], items[1]);
		}
		List<String> rel = new ArrayList<>();
		for (int i = 0; i < m.size(); ++i) {
			String s = start, t = m.get(start);
			rel.add(s + "-" + t);
			start = t;
		}
		return rel;
	}

	public int largestRectangleArea(int[] height) {
		if (height.length == 0)
			return 0;
		Stack<Integer> stk = new Stack<Integer>();
		int maxArea = 0;
		for (int i = 0; i < height.length; i++) {
			if (stk.isEmpty() || height[i] >= stk.peek())
				stk.push(height[i]);
			else {
				int count = 0;
				while (!stk.isEmpty() && stk.peek() > height[i]) {
					int t = stk.pop();
					count++;
					maxArea = Math.max(maxArea, t * count);
				}
				for (int j = 0; j <= count; j++) {
					stk.push(height[i]);
				}

			}
		}
		int count = 0;
		while (!stk.isEmpty()) {
			int t = stk.pop();
			count++;
			maxArea = Math.max(maxArea, t * count);

		}
		return maxArea;
	}

	public int largestRectangleArea2(int[] height) {
		int[] h = Arrays.copyOf(height, height.length + 1);
		Stack<Integer> stk = new Stack<Integer>();
		int maxArea = 0;
		int i = 0;
		while (i < h.length) {
			if (stk.isEmpty() || h[i] >= h[stk.peek()])
				stk.push(i++);
			else {
				int top = stk.pop();
				int length = stk.isEmpty() ? i : i - stk.peek() - 1;
				maxArea = Math.max(maxArea, h[top] * length);
			}
		}
		return maxArea;
	}

	public ListNode mergeKLists2(List<ListNode> lists) {
		if (lists.size() == 0)
			return null;
		ListNode res = mergeKLists2Util(lists, 0, lists.size() - 1);
		return res;
	}

	public ListNode mergeKLists2Util(List<ListNode> lists, int beg, int end) {
		if (beg >= end)
			return lists.get(beg);
		int mid = (beg + end) / 2;
		ListNode firstHalf = mergeKLists2Util(lists, beg, mid);
		ListNode secondHalf = mergeKLists2Util(lists, mid + 1, end);
		ListNode res = mergeTwoSortedLists2(firstHalf, secondHalf);
		return res;
	}

	public ListNode mergeTwoSortedLists2(ListNode first, ListNode second) {
		if (first == null || second == null)
			return first == null ? second : first;
		ListNode dummy = new ListNode(0);
		ListNode pre = dummy;
		while (first != null && second != null) {
			if (first.val < second.val) {
				pre.next = first;
				first = first.next;
			} else {
				pre.next = second;
				second = second.next;
			}
			pre = pre.next;
		}
		if (first != null)
			pre.next = first;
		if (second != null)
			pre.next = second;
		return dummy.next;
	}

	public static void solvex(char[][] board) {
		int m = board.length;
		if (m == 0)
			return;
		int n = board[0].length;
		for (int i = 0; i < m; i++) {
			dfs(i, 0, board);
			dfs(i, n - 1, board);
		}

		for (int i = 1; i < n - 1; i++) {
			dfs(0, i, board);
			dfs(m - 1, i, board);
		}

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 'O')
					board[i][j] = 'X';
				if (board[i][j] == '#')
					board[i][j] = 'O';
			}
		}

		for (int i = 0; i < m; i++) {
			System.out.println(Arrays.toString(board[i]));
		}
	}

	public static void dfs(int i, int j, char[][] board) {
		int m = board.length;
		int n = board[0].length;
		if (i >= m || i < 0 || j >= n || j < 0 || board[i][j] != 'O')
			return;
		board[i][j] = '#';
		dfs(i - 1, j, board);
		dfs(i + 1, j, board);
		dfs(i, j - 1, board);
		dfs(i, j + 1, board);
	}

	public static void solvex1(char[][] board) {
		int m = board.length;
		if (m == 0)
			return;
		int n = board[0].length;

		for (int i = 0; i < m; i++) {
			bfs(i, 0, board);
			bfs(i, n - 1, board);
		}

		for (int i = 1; i < n - 1; i++) {
			bfs(0, i, board);
			bfs(m - 1, i, board);
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

	public static void bfs(int i, int j, char[][] board) {
		int m = board.length;
		int n = board[0].length;
		Queue<int[]> que = new LinkedList<int[]>();

		if (board[i][j] == 'O') {
			que.add(new int[] { i, j });
		}
		while (!que.isEmpty()) {
			int[] pos = que.remove();
			int x = pos[0];
			int y = pos[1];
			board[x][y] = '#';
			if (x - 1 >= 0 && board[x - 1][y] == 'O') {
				que.add(new int[] { x - 1, y });
				board[x - 1][y] = '#';
			}
			if (x + 1 < m && board[x + 1][y] == 'O') {
				que.add(new int[] { x + 1, y });
				board[x + 1][y] = '#';
			}
			if (y - 1 >= 0 && board[x][y - 1] == 'O') {
				que.add(new int[] { x, y - 1 });
				board[x][y - 1] = '#';
			}
			if (y + 1 < n && board[x][y + 1] == 'O') {
				que.add(new int[] { x, y + 1 });
				board[x][y + 1] = '#';
			}
		}
	}

	// Find the smallest positive integer value that cannot be represented as
	// sum of any subset of a given array
	public static int findSmallest(int[] A) {
		int res = 1;
		int n = A.length;
		for (int i = 0; i < n && A[i] <= res; i++) {
			res += A[i];
		}
		return res;
	}

	public static RandomTreeNode cloneTree(RandomTreeNode root) {
		if (root == null)
			return null;
		HashMap<RandomTreeNode, RandomTreeNode> map = new HashMap<RandomTreeNode, RandomTreeNode>();
		RandomTreeNode clone = copyLeftRight(root, map);
		copyRandom(root, clone, map);
		return clone;
	}

	public static RandomTreeNode copyLeftRight(RandomTreeNode root,
			HashMap<RandomTreeNode, RandomTreeNode> map) {
		if (root == null)
			return null;
		RandomTreeNode clone = new RandomTreeNode(root.val);
		map.put(root, clone);
		clone.left = copyLeftRight(root.left, map);
		clone.right = copyLeftRight(root.right, map);
		return clone;
	}

	public static void copyRandom(RandomTreeNode root, RandomTreeNode clone,
			HashMap<RandomTreeNode, RandomTreeNode> map) {
		if (root == null)
			return;
		clone.random = map.get(root.random);
		copyRandom(root.left, clone.left, map);
		copyRandom(root.random, clone.right, map);
	}

	public static void printInorder(RandomTreeNode root) {
		if (root == null)
			return;
		printInorder(root.left);
		System.out.print("[" + root.val + " ");
		if (root.random == null)
			System.out.print("Null],");
		else
			System.out.println(root.random.val + "],");

		printInorder(root.right);
	}

	// Count all possible walks from a source to a destination with exactly k
	// edges
	public static int countwalksDp(int[][] graph, int u, int v, int k) {
		int V = graph.length;
		// // Table to be filled up using DP. The value count[i][j][e] will
		// store count of possible walks from i to j with exactly k edges
		int[][][] count = new int[V][V][k + 1];
		// for edges from 0 to k
		for (int e = 0; e <= k; e++) {
			for (int i = 0; i < V; i++) {// for source from 0 to V
				for (int j = 0; j < V; j++) { // for des from 0 to V
					if (e == 0 && i == j)
						count[i][j][e] = 1;
					else if (e == 1 && graph[i][j] == 1)
						count[i][j][e] = 1;
					else if (e > 1) {
						for (int l = 0; l < V; l++) {
							if (graph[i][l] == 1) {// i-->l-->j
								count[i][j][e] += count[l][j][e - 1];
							}
						}
					}
				}
			}
		}
		return count[u][v][k];
	}

	// A naive recursive function to count walks from u to v with k edges
	public static int countwalks(int graph[][], int u, int v, int k) {
		int V = graph.length;
		// Base cases
		if (k == 0 && u == v)
			return 1;
		if (k == 1 && graph[u][v] == 1)
			return 1;
		if (k <= 0)
			return 0;

		// Initialize result
		int count = 0;

		// Go to all adjacents of u and recur
		for (int i = 0; i < V; i++)
			if (graph[u][i] == 1) // Check if is adjacent of u
				count += countwalks(graph, i, v, k - 1);

		return count;
	}

	// Shortest path with exactly k edges in a directed and weighted graph

	public static int shortestPathRecur(int[][] graph, int u, int v, int k) {
		int V = graph.length;
		if (u == v && k == 0)
			return 0;
		if (graph[u][v] != Integer.MAX_VALUE && k == 1)
			return graph[u][v];
		if (k <= 0)
			return Integer.MAX_VALUE;
		int res = Integer.MAX_VALUE;
		for (int i = 0; i < V; i++) {
			if (graph[u][i] != Integer.MAX_VALUE && u != i && v != i) {
				int resPartial = shortestPathRecur(graph, i, v, k - 1);
				if (resPartial != Integer.MAX_VALUE)
					res = Math.min(res, resPartial + graph[u][i]);
			}
		}
		return res;
	}

	public static int shortestPathDp(int[][] graph, int u, int v, int k) {
		int V = graph.length;
		// Table to be filled up using DP. The value sp[i][j][e] will store
		// weight of the shortest path from i to j with exactly k edges
		int[][][] dp = new int[V][V][k + 1];
		for (int e = 0; e <= k; e++) {
			for (int i = 0; i < V; i++) {
				for (int j = 0; j < V; j++) {
					dp[i][j][e] = Integer.MAX_VALUE;
					if (i == j && e == 0)
						dp[i][j][e] = 0;
					if (graph[i][j] != Integer.MAX_VALUE && e == 1)
						dp[i][j][e] = graph[i][j];
					if (e > 1) {
						for (int a = 0; a < V; a++) {
							if (graph[i][a] != Integer.MAX_VALUE && i != a
									&& j != a
									&& dp[a][j][e - 1] != Integer.MAX_VALUE) {
								dp[i][j][e] = Math.min(dp[i][j][e],
										dp[a][j][e - 1] + graph[i][a]);
							}
						}
					}
				}
			}
		}
		return dp[u][v][k];
	}

	public static int countDecoding(String s) {
		if (s.length() == 0 || s.length() == 1)
			return 1;
		int n = s.length();
		int count = 0;
		if (s.charAt(n - 1) != '0')
			count = countDecoding(s.substring(0, n - 1));
		if (s.charAt(n - 2) == '1' || s.charAt(n - 2) == '2'
				&& s.charAt(n - 1) < '7')
			count += countDecoding(s.substring(0, n - 2));
		return count;
	}

	public static int countDecodingDp(String s) {
		int n = s.length();
		int[] count = new int[n + 1];
		count[0] = count[1] = 1;
		for (int i = 2; i <= n; i++) {
			if (s.charAt(i - 1) != '0')
				count[i] = count[i - 1];
			if (s.charAt(i - 2) == '1' || s.charAt(i - 2) == '2'
					&& s.charAt(i - 1) < '7')
				count[i] += count[i - 2];
		}
		return count[n];
	}

	// Connect n ropes with minimum cost
	public static int minCost(int[] ropes) {
		if (ropes.length == 0)
			return 0;
		if (ropes.length == 1)
			return ropes[0];

		PriorityQueue<Integer> que = new PriorityQueue<Integer>();
		for (int i : ropes) {
			que.add(i);
		}
		int cost = 0;

		while (que.size() > 1) {
			int first_min = que.poll();
			int second_min = que.poll();
			cost += first_min + second_min;
			que.add(first_min + second_min);
		}
		return cost;
	}

	// Search in an almost sorted array
	// Basically the element arr[i] can only be swapped with either arr[i+1] or
	// arr[i-1].
	public static int binarySearch(int[] A, int target) {
		int n = A.length;
		return binarySearchUtil(A, 0, n - 1, target);
	}

	public static int binarySearchUtil(int[] A, int beg, int end, int target) {
		if (beg > end)
			return -1;
		int mid = (beg + end) / 2;
		if (A[mid] == target)
			return mid;
		if (mid > beg && A[mid - 1] == target)
			return mid - 1;
		if (mid < end && A[mid + 1] == target)
			return mid + 1;
		if (A[mid] > target)
			return binarySearchUtil(A, beg, mid - 2, target);
		return binarySearchUtil(A, mid + 2, end, target);
	}

	// Check if two nodes are cousins in a Binary Tree
	public static boolean isCousin(TreeNode root, TreeNode node1, TreeNode node2) {
		if (root == null)
			return false;
		int l1 = getNodeLevel(root, node1, 1);
		int l2 = getNodeLevel(root, node2, 1);

		if (l1 == l2 && !isSibling(root, node1, node2))
			return true;
		else
			return false;
	}

	public static int getNodeLevel(TreeNode root, TreeNode node, int level) {
		if (root == null)
			return 0;
		if (root == node)
			return level;
		int left = getNodeLevel(root.left, node, level + 1);
		if (left != 0)
			return left;
		return getNodeLevel(root.right, node, level + 1);
	}

	public static boolean isSibling(TreeNode root, TreeNode node1,
			TreeNode node2) {
		if (root == null)
			return false;
		// return (root.left==node1&&root.right==node2)||
		// (root.left==node2&&root.right==node1)||
		// isSibling(root.left,node1, node2)||isSibling(root.right,
		// node1,node2);
		if (root == node1 || root == node2)
			return false;
		if (root.left == node1 && root.right == node2 || root.left == node2
				&& root.right == node1)
			return true;
		return isSibling(root.left, node1, node2)
				|| isSibling(root.right, node1, node2);
	}

	public static int firstRepeating(int[] A) {
		if (A.length < 2)
			return Integer.MAX_VALUE;
		HashSet<Integer> set = new HashSet<Integer>();
		int min = -1;
		for (int i = A.length - 1; i >= 0; i--) {
			if (set.contains(A[i]))
				min = i;
			else
				set.add(A[i]);
		}
		if (min == -1)
			return Integer.MAX_VALUE;
		return A[min];
	}

	public static void findCommon(int arr1[], int arr2[], int arr3[]) {
		int n1 = arr1.length;
		int n2 = arr2.length;
		int n3 = arr3.length;
		int i = 0, j = 0, k = 0;
		while (i < n1 && j < n2 && k < n3) {
			if (arr1[i] == arr2[j] && arr2[j] == arr3[k]) {
				System.out.print(arr1[i] + " ");
				i++;
				j++;
				k++;
			} else if (arr1[i] < arr2[j])
				i++;
			else if (arr2[j] < arr3[k])
				j++;
			else
				k++;
		}
		System.out.println();
	}

	public int totalNQueens(int n) {
		if (n <= 0)
			return 0;
		int[] res = { 0 };
		int[] loc = new int[n];
		dfsNQueens(0, n, loc, res);
		return res[0];
	}

	public void dfsNQueens(int cur, int n, int[] loc, int[] res) {
		if (cur == n) {
			res[0]++;
			return;
		}
		for (int i = 0; i < n; i++) {
			loc[cur] = i;
			if (isValid(loc, cur))
				dfsNQueens(cur + 1, n, loc, res);
		}
	}

	public boolean isValid(int[] loc, int cur) {
		for (int i = 0; i < cur; i++) {
			if (loc[i] == loc[cur]
					|| Math.abs(loc[cur] - loc[i]) == Math.abs(cur - i))
				return false;
		}
		return true;
	}

	public List<String[]> solveNQueens(int n) {
		List<String[]> res = new ArrayList<String[]>();
		int[] loc = new int[n];
		dfsNQueens(0, n, loc, res);
		return res;
	}

	public void dfsNQueens(int cur, int n, int[] loc, List<String[]> res) {
		if (cur == n) {
			printBoard(n, loc, res);
			return;
		}
		for (int i = 0; i < n; i++) {
			loc[cur] = i;
			if (isValid(cur, loc))
				dfsNQueens(cur + 1, n, loc, res);
		}
	}

	public boolean isValid(int cur, int[] loc) {
		for (int i = 0; i < cur; i++) {
			if (loc[i] == loc[cur] || Math.abs(loc[cur] - loc[i]) == cur - i)
				return false;
		}
		return true;
	}

	public void printBoard(int n, int[] loc, List<String[]> res) {
		String[] str = new String[n];
		for (int i = 0; i < n; i++) {
			String row = "";
			for (int j = 0; j < n; j++) {
				if (loc[i] == j)
					row += 'Q';
				else
					row += '.';
			}
			str[i] = row;
		}
		res.add(str);
	}

	public void solveSudoku(char[][] board) {
		List<int[]> empty = new ArrayList<int[]>();
		for (int i = 0; i < 9; i++) {
			for (int j = 0; j < 9; j++) {
				if (board[i][j] == '.') {
					int[] dot = { i, j };
					empty.add(dot);
				}
			}
		}
		dfsSudoku(0, empty, board);
	}

	public boolean dfsSudoku(int cur, List<int[]> empty, char[][] board) {
		if (cur == empty.size())
			return true;
		int row = empty.get(cur)[0];
		int col = empty.get(cur)[1];

		for (int i = 1; i <= 9; i++) {
			if (isValidSudoku(i, row, col, board)) {
				board[row][col] = (char) ('0' + i);
				if (dfsSudoku(cur + 1, empty, board))
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
			int b_row = 3 * (row / 3) + i / 3;
			int b_col = 3 * (col / 3) + i % 3;
			if (board[b_row][b_col] == '0' + val)
				return false;
		}
		return true;
	}

	private boolean isValidSudoku(char[][] board, int i, int j, char c) {

		// check column
		for (int row = 0; row < 9; row++) {
			if (board[row][j] == c) {
				return false;
			}
		}

		// check row
		for (int col = 0; col < 9; col++) {
			if (board[i][col] == c) {
				return false;
			}
		}

		// check block
		for (int row = i / 3 * 3; row < i / 3 * 3 + 3; row++) {
			for (int col = j / 3 * 3; col < j / 3 * 3 + 3; col++) {
				if (board[row][col] == c) {
					return false;
				}
			}
		}
		return true;
	}

	public List<String> wordBreak0921(String s, Set<String> dict) {
		List<String> res = new ArrayList<String>();
		int n = s.length();
		boolean[][] dp = new boolean[n][n + 1];
		for (int i = n - 1; i >= 0; i--) {
			for (int j = i + 1; j <= n; j++) {
				String sub = s.substring(i, j);
				if (dict.contains(sub) && j == n) {
					dp[i][j - 1] = true;
					dp[i][n] = true;
				} else if (dict.contains(sub) && j < n && dp[j][n]) {
					dp[i][j - 1] = true;
					dp[i][n] = true;
				}
			}
		}
		if (!dp[0][n])
			return res;
		wordBreakUtil(s, 0, "", dp, res);
		return res;
	}

	public void wordBreakUtil(String s, int cur, String sol, boolean[][] dp,
			List<String> res) {
		if (cur == s.length()) {
			res.add(sol);
		}
		for (int i = cur; i < s.length(); i++) {
			if (dp[cur][i]) {
				String sub = "";
				if (i < s.length() - 1)
					sub = sol + s.substring(cur, i + 1) + " ";
				else
					sub = sol + s.substring(cur, i + 1);
				wordBreakUtil(s, i + 1, sub, dp, res);
			}
		}
	}

	public List<List<String>> partition(String s) {
		List<List<String>> res = new ArrayList<List<String>>();
		List<String> sol = new ArrayList<String>();
		partitionUtil(0, s, sol, res);
		return res;
	}

	public void partitionUtil(int cur, String s, List<String> sol,
			List<List<String>> res) {
		if (cur == s.length()) {
			List<String> out = new ArrayList<String>(sol);
			res.add(out);
		}

		for (int i = cur; i < s.length(); i++) {
			if (isPalindrome2(s.substring(cur, i + 1))) {
				sol.add(s.substring(cur, i + 1));
				partitionUtil(i + 1, s, sol, res);
				sol.remove(sol.size() - 1);
			}
		}
	}

	public boolean isPalindrome2(String s) {
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

	public int minCut(String s) {
		int n = s.length();
		int[] cut = new int[n + 1];
		boolean[][] p = new boolean[n][n];
		for (int i = 0; i <= n; i++) {
			cut[i] = n - i;
		}

		for (int i = n - 1; i >= 0; i++) {
			for (int j = i; j < n; j++) {
				if (s.charAt(i) == s.charAt(j)
						&& (j - i < 2 || p[i + 1][j - 1])) {
					p[i][j] = true;
					cut[i] = Math.min(cut[i], cut[j + 1] + 1);
				}
			}
		}
		return cut[0] - 1;
	}

	// '.' Matches any single character.
	// '*' Matches zero or more of the preceding element.

	public boolean isMatch(String s, String p) {
		// if(s.length()==0)
		// return check(p);
		// if(p.length()==0)
		// return false;
		// char s1=s.charAt(0);
		// char p1=p.charAt(0);
		// char p2='0';
		// if(p.length()>1)
		// p2=p.charAt(1);
		// if(p2=='*'){
		// if(p1==s1||p1=='.')
		// return isMatch(s,p.substring(2))||isMatch(s.substring(1),p);
		// else
		// return isMatch(s,p.substring(2));
		// }
		// else{
		// if(p1==s1||p1=='.')
		// return isMatch(s.substring(1),p.substring(1));
		// else
		// return false;
		// }
		// }

		// public boolean check(String s){
		// if(s.length()%2!=0)
		// return false;
		// for(int i=1;i<s.length();i+=2){
		// if(s.charAt(i)!='*')
		// return false;
		// }
		// return true;

		int lenS = s.length();
		int lenP = p.length();

		if (lenP == 0)
			return lenS == 0;
		if (lenP == 1) {
			if (s.length() == 1
					&& (s.charAt(0) == p.charAt(0) || p.charAt(0) == '.'))
				return true;
			else
				return false;
		}
		if (p.charAt(1) != '*') {
			if (s.length() > 0
					&& (s.charAt(0) == p.charAt(0) || p.charAt(0) == '.'))
				return isMatch(s.substring(1), p.substring(1));
			return false;
		} else {
			while (s.length() > 0
					&& (s.charAt(0) == p.charAt(0) || p.charAt(0) == '.')) {
				if (isMatch(s, p.substring(2)))
					return true;
				s = s.substring(1);
			}
			return isMatch(s, p.substring(2));
		}
	}

	public int divide(int dividend, int divisor) {
		boolean neg = (dividend < 0 && divisor > 0)
				|| (dividend > 0 && divisor < 0);
		long a = Math.abs((long) dividend);
		long b = Math.abs((long) divisor);
		int ans = 0;
		while (a >= b) {
			int shift = 0;
			while ((b << shift) <= a) {
				shift++;
			}
			ans += 1 << (shift - 1);
			a -= b << (shift - 1);
		}
		return neg ? -ans : ans;
	}

	// uodated overflow

	public int divide2(int dividend, int divisor) {
		boolean neg = (dividend < 0 && divisor > 0)
				|| (dividend > 0 && divisor < 0);
		boolean overflow = false;

		long dvd = Math.abs((long) dividend);
		long dvs = Math.abs((long) divisor);

		int res = 0;
		while (dvd >= dvs) {
			int shift = 0;
			while (dvd >= (dvs << shift)) {
				shift++;
			}
			dvd -= dvs << (shift - 1);
			if (Integer.MAX_VALUE - (1 << (shift - 1)) < res) {
				overflow = true;
				break;
			}
			res += 1 << (shift - 1);
		}
		if (neg) {
			if (overflow)
				return Integer.MIN_VALUE;
			return -res;
		} else {
			if (overflow)
				return Integer.MAX_VALUE;
			return res;
		}
	}
	
	public List<String> fullJustify(String[] words, int maxWidth) {
		List<String> res=new ArrayList<String>();
		int n=words.length;
		int i=0;
		while(i<n){
			int start=i, sum=0;
			while(i<n&&sum+words[i].length()<=maxWidth){
				sum+=words[i].length()+1;
				i++;
			}
			int end=i-1;
			int intervals=end-start;
			int spaces=0, extra=0;
			if(intervals>0){
				spaces=(maxWidth-(sum-intervals-1))/intervals;
				extra=(maxWidth-(sum-intervals-1))%intervals;
			}
			StringBuilder sb=new StringBuilder(words[start]);
			for(int j=start+1;j<=end;j++){
				if(i==words.length)//last line
					sb.append(" ");
				else{
					for(int s=spaces;s>0;s--)
						sb.append(" ");
					if(extra-->0)
						sb.append(" ");
				}
				sb.append(words[j]);
			}
			int left=maxWidth-sb.length();
			while(left-->0){
				sb.append(" ");
			}
			res.add(sb.toString());
		}
		return res;
	}

	public int ladderLength(String start, String end, Set<String> dict) {
		Queue<String> que = new LinkedList<String>();
		Set<String> visited = new HashSet<String>();
		int curlevel = 0;
		int nextlevel = 0;
		que.add(start);
		curlevel++;
		visited.add(start);
		int steps = 1;

		while (!que.isEmpty()) {
			String top = que.remove();
			curlevel--;
			if (top.equals(end))
				return steps;
			char[] ch = top.toCharArray();
			for (int i = 0; i < ch.length; i++) {
				char t = ch[i];
				for (char c = 'a'; c <= 'z'; c++) {
					if (t != c) {
						ch[i] = c;
						String s = new String(ch);
						if (dict.contains(s) && !visited.contains(s)) {
							visited.add(s);
							que.add(s);
							nextlevel++;
						}
					}
				}
				ch[i] = t;
			}
			if (curlevel == 0) {
				curlevel = nextlevel;
				nextlevel = 0;
				steps++;
			}
		}
		return 0;
	}

	public int ladderLength(String start, String end, HashSet<String> dict) {
		Queue<String> que = new LinkedList<String>();
		que.offer(start);
		dict.remove(start);
		int steps = 1;

		while (!que.isEmpty()) {
			int count = que.size();
			for (int i = 0; i < count; i++) {
				String cur = que.poll();
				for (char c = 'a'; c <= 'z'; c++) {
					for (int j = 0; j < cur.length(); j++) {
						if (c != cur.charAt(j)) {
							String tmp = replace(cur, j, c);
							if (tmp.equals(end))
								return steps + 1;
							if (dict.contains(tmp)) {
								que.offer(tmp);
								dict.remove(tmp);
							}
						}
					}
				}
			}
			steps++;
		}
		return 0;
	}

	private String replace(String s, int j, char c) {
		char[] chars = s.toCharArray();
		chars[j] = c;
		return new String(chars);
	}

	public boolean isNumber(String s) {
		s = s.trim();
		if (s.length() == 0)
			return false;
		int i = 0;
		int n = s.length();
		if (s.charAt(0) == '+' || s.charAt(i) == '-')
			i++;
		boolean num = false;
		boolean dot = false;
		boolean exp = false;
		while (i < n) {
			char c = s.charAt(i);
			if (Character.isDigit(c))
				num = true;
			else if (c == '.') {
				if (exp || dot)
					return false;
				dot = true;
			} else if (c == 'e' || c == 'E') {
				if (exp || !num)
					return false;
				exp = true;
				num = false;
			} else if (c == '+' || c == '-') {
				if (s.charAt(i - 1) != 'e')
					return false;
			} else
				return false;
			i++;
		}
		return num;
	}

	public double findMedianSortedArrays(int A[], int B[]) {
		int m = A.length;
		int n = B.length;
		if ((m + n) % 2 == 0)
			return (findKth(A, 0, m, B, 0, n, (m + n) / 2) + findKth(A, 0, m,
					B, 0, n, (m + n) / 2 + 1)) / 2.0;
		else
			return findKth(A, 0, m, B, 0, n, (m + n) / 2 + 1);
	}

	public double findKth(int[] A, int aoffset, int m, int[] B, int boffset,
			int n, int k) {
		if (m > n)
			return findKth(B, boffset, n, A, aoffset, m, k);
		if (m == 0)
			return B[k - 1];
		if (k == 1)
			return Math.min(A[aoffset], B[boffset]);
		int pa = Math.min(m, k / 2);
		int pb = k - pa;
		if (A[aoffset + pa - 1] < B[boffset + pb - 1])
			return findKth(A, aoffset + pa, m - pa, B, boffset, n, k - pa);
		else
			return findKth(A, aoffset, m, B, boffset + pb, n - pb, k - pb);
	}

	public boolean isMatch2(String s, String p) {
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
				i = ++sp;
				j = star + 1;
			} else
				return false;
		}
		while (j < p.length() && p.charAt(j) == '*')
			j++;
		return j == p.length();
	}

	// 假设数组为a[]，直接利用动归来求解，考虑到可能存在负数的情况，我们用Max来表示以a结尾的最大连续子串的乘积值，用Min表示以a结尾的最小的子串的乘积值，那么状态转移方程为：
	// Max=max{a, Max[i-1]*a, Min[i-1]*a};
	// Min=min{a, Max[i-1]*a, Min[i-1]*a};
	// 初始状态为Max[1]=Min[1]=a[1]。
	// Max is the max product ending at a, Min is the min product ending at a;

	public int maxProduct(int[] A) {
		int n = A.length;
		int[] maxProd = new int[n];
		int[] minProd = new int[n];
		maxProd[0] = minProd[0] = A[0];
		int res = A[0];

		for (int i = 1; i < n; i++) {
			maxProd[i] = Math.max(A[i],
					Math.max(maxProd[i - 1] * A[i], minProd[i - 1] * A[i]));
			minProd[i] = Math.min(A[i],
					Math.min(maxProd[i - 1] * A[i], minProd[i - 1] * A[i]));
			res = Math.max(res, maxProd[i]);
		}
		return res;
	}

	public int maxProduct2(int[] A) {
		int maxProduct = 1;
		int minProduct = 1;
		int maxCurrent = 1;
		int minCurrent = 1;

		for (int i = 0; i < A.length; i++) {
			maxCurrent *= A[i];
			minCurrent *= A[i];

			if (maxCurrent > maxProduct)
				maxProduct = maxCurrent;
			if (minCurrent > maxProduct)
				maxProduct = minCurrent;
			if (maxCurrent < minProduct)
				minProduct = maxProduct;
			if (minCurrent < minProduct)
				minProduct = minCurrent;
			if (minCurrent > maxCurrent) {
				int t = maxCurrent;
				maxCurrent = minCurrent;
				minCurrent = t;
			}
			if (maxCurrent < 1)
				maxCurrent = 1;
		}
		return maxProduct;
	}

	public int maxProduct3(int[] A) {
		int R = 0;
		int r = 0;
		int Max = A[0], Min = A[0];
		int res = A[0];
		Pair pair = new Pair(0, 0);
		for (int i = 1; i < A.length; i++) {
			int t0 = A[i] * Max;
			int t1 = A[i] * Min;
			if (t0 > t1) {
				Max = t0;
				Min = t1;
			} else {
				int t = R;
				R = r;
				r = t;
				Max = t1;
				Min = t0;
			}
			if (Max < A[i]) {
				Max = A[i];
				R = i;
			}
			if (Min > A[i]) {
				Min = A[i];
				r = i;
			}

			if (res < Max) {
				res = Max;
				pair.first = R;
				pair.second = i;
			}
		}
		return res;
	}

	public int maxProduct4(int[] A) {
		int curMax = 1;
		int curMin = 1;
		int allMax = Integer.MIN_VALUE;
		for (int i = 0; i < A[i]; i++) {
			if (A[i] >= 0) {
				curMax = curMax <= 0 ? A[i] : curMax * A[i];
				curMin = curMin * A[i];
			} else {
				int tmp = curMax;
				curMax = Math.max(curMin * A[i], A[i]);
				curMin = Math.min(tmp * A[i], A[i]);
			}
			allMax = Math.max(allMax, curMax);
		}
		return allMax;
	}

	public List<List<String>> findLadders(String start, String end,
			Set<String> dict) {
		List<List<String>> res = new ArrayList<List<String>>();
		Queue<String> curLevel = new LinkedList<String>();
		HashMap<String, List<String>> map = new HashMap<String, List<String>>();
		boolean exist = false;
		Set<String> visited = new HashSet<String>();

		curLevel.offer(start);
		dict.add(end);
		visited.add(start);

		while (!curLevel.isEmpty()) {
			Set<String> toBuild = new HashSet<String>();
			Queue<String> nextLevel = new LinkedList<String>();
			while (!curLevel.isEmpty()) {
				String s = curLevel.poll();
				ArrayList<String> neighbor = new ArrayList<String>();
				char[] word = s.toCharArray();
				for (int i = 0; i < s.length(); i++) {
					char t = word[i];
					for (char c = 'a'; c <= 'z'; c++) {
						if (word[i] != c) {
							word[i] = c;
							String st = new String(word);
							if (dict.contains(st) && !visited.contains(st)) {
								neighbor.add(st);
								if (toBuild.add(st))
									nextLevel.offer(st);
							}
							exist = exist || st.equals(end);
						}
					}
					word[i] = t;
				}
				map.put(s, neighbor);
			}
			visited.addAll(toBuild);
			if (exist)
				break;
			curLevel = nextLevel;
		}
		if (exist)
			dfsLadder(start, end, map, new ArrayList<String>(), res);
		return res;
	}

	public void dfsLadder(String start, String end,
			HashMap<String, List<String>> map, ArrayList<String> sol,
			List<List<String>> res) {
		if (start.equals(end)) {
			ArrayList<String> out = new ArrayList<String>(sol);
			out.add(start);
			res.add(out);
			return;
		}
		if (!map.containsKey(start))
			return;
		if (map.containsKey(start)) {
			sol.add(start);
			List<String> list = map.get(start);
			for (int i = 0; i < list.size(); i++) {
				dfsLadder(list.get(i), end, map, sol, res);
			}
			sol.remove(sol.size() - 1);
		}
	}

	public List<List<String>> findLadders2(String start, String end,
			Set<String> dict) {
		List<List<String>> res = new ArrayList<List<String>>();
		if (dict.size() == 0)
			return res;
		HashMap<String, List<List<String>>> map = new HashMap<String, List<List<String>>>();
		List<List<String>> paths = new ArrayList<List<String>>();
		List<String> path = new ArrayList<String>();
		path.add(start);
		paths.add(path);
		map.put(start, paths);
		Queue<String> que = new LinkedList<String>();
		que.add(start);

		while (!que.isEmpty()) {
			String str = que.remove();
			List<List<String>> ps = map.get(str);
			if (str.equals(end))
				return ps;
			char[] ch = str.toCharArray();
			for (int i = 0; i < ch.length; i++) {
				char t = ch[i];
				for (char c = 'a'; c <= 'z'; c++) {
					if (t != c) {
						ch[i] = c;
						String s = new String(ch);
						if (dict.contains(s)) {
							List<List<String>> toAdd = new ArrayList<List<String>>();
							if (!map.containsKey(s)) {
								que.add(s);
								for (List<String> p : ps) {
									List<String> np = new ArrayList<String>(p);
									np.add(s);
									toAdd.add(np);
								}
								map.put(s, toAdd);
							} else if (map.get(s).get(0).size() == ps.get(0)
									.size() + 1) {
								for (List<String> p : ps) {
									List<String> np = new ArrayList<String>(p);
									np.add(s);
									toAdd.add(np);
								}
								toAdd.addAll(map.get(s));
								map.put(s, toAdd);
							}
						}
					}
				}
				ch[i] = t;
			}
		}
		return res;
	}

	public static boolean isIsomorphic(String s1, String s2) {
		if (s1.length() != s2.length())
			return false;
		int len = s1.length();
		if (len == 1)
			return true;
		HashMap<Character, Character> map1 = new HashMap<Character, Character>();
		HashMap<Character, Character> map2 = new HashMap<Character, Character>();
		for (int i = 0; i < len; i++) {
			char c1 = s1.charAt(i);
			char c2 = s2.charAt(i);
			if (map1.containsKey(c1)) {
				if (map1.get(c1) != c2)
					return false;
			}
			if (map2.containsKey(c2)) {
				if (map2.get(c2) != c1)
					return false;
			}
			map1.put(c1, c2);
			map2.put(c2, c1);
		}
		return true;
	}

	// word distance LinkedIn
	public static int minDistStrings(String[] strs, String str1, String str2) {
		if (strs.length < 2)
			return -1;
		int lastPos = -1;
		int minDist = Integer.MAX_VALUE;
		for (int i = 0; i < strs.length; i++) {
			if (strs[i].equals(str1) || strs[i].equals(str2)) {
				lastPos = i;
				break;
			}
		}

		for (int i = lastPos + 1; i < strs.length; i++) {
			if (strs[i].equals(str1) || strs[i].equals(str2)) {
				if (!strs[i].equals(strs[lastPos]) && i - lastPos < minDist)
					minDist = i - lastPos;
				lastPos = i;
			}
		}
		return minDist == Integer.MAX_VALUE ? -1 : minDist;
	}

	public static int minDistStringsRelative(String[] strs, String str1,
			String str2) {
		if (strs.length < 2)
			return -1;
		int lastPos = -1;
		int minDist = Integer.MAX_VALUE;
		for (int i = 0; i < strs.length; i++) {
			if (strs[i].equals(str1)) {
				lastPos = i;
				break;
			}
		}
		if (lastPos == -1)
			return -1;

		for (int i = lastPos + 1; i < strs.length; i++) {
			if (strs[i].equals(str1) || strs[i].equals(str2)) {
				if (strs[i].equals(str2) && i - lastPos < minDist)
					minDist = i - lastPos;
				else
					lastPos = i;
			}
		}
		return minDist == Integer.MAX_VALUE ? -1 : minDist;
	}

	public static int InfluencerFinder(int[][] matrix) {
		int n = matrix.length;

		for (int i = 0; i < n; i++) {
			boolean is_influencer = true;
			for (int j = 0; j < n; j++) {
				if (i == j)
					continue;
				if (matrix[i][j] == 1 || matrix[j][i] == 0) {
					is_influencer = false;
					break;
				}
			}
			if (is_influencer)
				return i;
		}
		return -1;
	}

	public static TreeNode flipDown(TreeNode root) {
		if (root == null)
			return null;
		if (root.left == null && root.right == null)
			return root;
		TreeNode node = flipDown(root.left);
		root.left.left = root.right;
		root.left.right = root;
		root.left = root.right = null;
		return node;

	}

	public static String serializeBTree(TreeNode root) {
		if (root == null)
			return "";
		String res = "";
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		que.add(root);
		while (!que.isEmpty()) {
			TreeNode top = que.remove();
			if (top == null)
				res += "# ";
			else {
				res += top.val + " ";
				que.add(top.left);
				que.add(top.right);
			}
		}
		return res;
	}

	public static String serializePreorder0(TreeNode root) {
		if (root == null)
			return "# ";
		return root.val + " " + serializePreorder(root.left)
				+ serializePreorder(root.right);
	}

	public static String serializePreorder(TreeNode root) {
		// if(root==null)
		// return "# ";
		// return
		// root.val+" "+serializePreorder(root.left)+serializePreorder(root.right);
		String[] res = { "" };
		serializePreorderUtil(root, res);
		return res[0];
	}

	public static void serializePreorderUtil(TreeNode root, String[] res) {
		if (root == null) {
			res[0] += "# ";
		} else {
			res[0] += root.val + " ";
			serializePreorderUtil(root.left, res);
			serializePreorderUtil(root.right, res);
		}
	}

	public static TreeNode deserializeBTree(String res) {
		String[] tokens = res.trim().split(" ");
		int[] index = { 0 };
		return deserializeBTreeUtil(tokens, index);

	}

	public static TreeNode deserializeBTreeUtil(String[] tokens, int[] index) {
		if (index[0] > tokens.length)
			return null;
		if (tokens[index[0]].equals("#")) {
			index[0]++;
			return null;
		}

		int val = Integer.parseInt(tokens[index[0]]);
		TreeNode root = new TreeNode(val);
		index[0]++;
		root.left = deserializeBTreeUtil(tokens, index);
		root.right = deserializeBTreeUtil(tokens, index);
		return root;
	}

	public static List<Integer> printLeaf(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		if (root == null)
			return res;
		printLeaf(root, res);
		return res;
	}

	public static void printLeaf(TreeNode root, List<Integer> res) {
		if (root == null)
			return;
		if (root.left == null && root.right == null)
			res.add(root.val);
		printLeaf(root.left, res);
		printLeaf(root.right, res);
	}

	public static boolean isBipartite(int[][] matrix, int src) {
		int n = matrix.length;
		int[] colorArr = new int[n];
		for (int i = 0; i < n; i++)
			colorArr[i] = -1;
		colorArr[src] = 1;

		Queue<Integer> que = new LinkedList<Integer>();
		que.offer(src);
		while (!que.isEmpty()) {
			int u = que.remove();

			for (int v = 0; v < n; v++) {
				if (matrix[u][v] == 1 && colorArr[v] == -1) {
					colorArr[v] = 1 - colorArr[u];
					que.offer(v);
				} else if (matrix[u][v] == 1 && colorArr[v] == colorArr[u])
					return false;
			}
		}
		return true;
	}

	// Minimum Cost Polygon Triangulation--recursion
	public static double minPolygonTirangulation(Point[] points) {
		int n = points.length;
		return minPolygonTirangulationUtil(points, 0, n - 1);
	}

	public static double minPolygonTirangulationUtil(Point[] points, int i,
			int j) {
		if (j < i + 2)
			return 0;
		double res = 0;
		for (int k = i + 1; k < j; k++) {
			res = Math.min(
					res,
					minPolygonTirangulationUtil(points, i, k)
							+ minPolygonTirangulationUtil(points, k, j)
							+ cost(points, i, j, k));
		}
		return res;
	}

	// Minimum Cost Polygon Triangulation --dp

	public static double minPolygonTriangulatuonDP(Point[] points) {
		int n = points.length;
		double[][] table = new double[n][n];

		for (int gap = 0; gap < n; gap++) {
			for (int i = 0, j = gap; j < n; i++, j++) {
				if (j < i + 2)
					table[i][j] = 0.0;
				else {
					table[i][j] = Double.MAX_VALUE;
					for (int k = i + 1; k < j; k++) {
						double val = table[i][k] + table[k][j]
								+ cost(points, i, j, k);
						if (table[i][j] > val)
							table[i][j] = val;
					}
				}
			}
		}
		return table[0][n - 1];
	}

	public static double cost(Point[] points, int i, int k, int j) {
		Point p1 = points[i], p2 = points[j], p3 = points[k];
		return dist(p1, p2) + dist(p2, p3) + dist(p3, p1);
	}

	public static double dist(Point p1, Point p2) {
		return Math.sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y)
				* (p1.y - p2.y));
	}

	// Find Height of Binary Tree represented by Parent array
	// A given array represents a tree in such a way that the array value gives
	// the parent node of that particular index.

	public static int findHeight(int[] parent) {
		int[] depth = new int[parent.length];
		for (int i = 0; i < parent.length; i++) {
			fillDepth(parent, i, depth);
		}
		int maxDep = depth[0];
		for (int i = 0; i < depth.length; i++)
			maxDep = Math.max(maxDep, depth[i]);
		return maxDep;
	}

	public static void fillDepth(int[] parent, int i, int[] depth) {
		// If depth[i] is already filled
		if (depth[i] != 0)
			return;
		// If node at index i is root
		if (parent[i] == -1) {
			depth[i] = 1;
			return;
		}
		// If depth of parent is not evaluated before, then evaluate
		// depth of parent first
		if (depth[parent[i]] == 0)
			fillDepth(parent, parent[i], depth);

		// Depth of this node is depth of parent plus 1
		depth[i] = depth[parent[i]] + 1;

	}

	// LinkedIn factor combination of number
	public static List<List<Integer>> printFactors(int target) {
		List<Integer> factors = new ArrayList<Integer>();
		for (int i = 2; i <= target / 2; i++) {
			factors.add(i);
		}
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		factorCompUtil(0, target, factors, sol, res);
		res.add(0, new ArrayList<Integer>(Arrays.asList(1, target)));
		return res;
	}

	public static void factorCompUtil(int cur, int target,
			List<Integer> factors, List<Integer> sol, List<List<Integer>> res) {
		if (target == 1) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
			return;
		}
		for (int i = cur; i < factors.size(); i++) {
			if (target >= factors.get(i) && target % factors.get(i) == 0) {
				sol.add(factors.get(i));
				factorCompUtil(i, target / factors.get(i), factors, sol, res);
				sol.remove(sol.size() - 1);
			}
		}
	}

	public static ComplexNode flattenList(ComplexNode head) {
		if (head == null)
			return null;
		ComplexNode tail = head;
		while (tail.next != null)
			tail = tail.next;
		ComplexNode cur = head;
		while (cur != tail) {
			if (cur.child != null) {
				tail.next = cur.child;
				ComplexNode node = cur.next;
				while (node.next != null) {
					node = node.next;
				}
				tail = node;
			}
			cur = cur.next;
		}
		return head;
	}

	public static int findMajority(int[] A) {
		if (A.length == 0)
			return Integer.MAX_VALUE;
		int majority = A[0];
		int count = 1;
		for (int i = 1; i < A.length; i++) {
			if (A[i] == majority)
				count++;
			else
				count--;
			if (count == 0) {
				majority = A[i];
				count = 1;
			}
		}

		count = 0;
		for (int i = 0; i < A.length; i++) {
			if (A[i] == majority)
				count++;
		}
		return count > A.length / 2 ? majority : Integer.MAX_VALUE;
	}

	public static int deepLevelSum(List<Object> list) {
		return deepLevelSumUtil(list, 1);
	}

	public static int deepLevelSumUtil(List<Object> list, int level) {
		int sum = 0;
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i) instanceof Integer)
				sum += level * (int) list.get(i);
			else
				sum += deepLevelSumUtil((List<Object>) list.get(i), level + 1);
		}
		return sum;
	}

	// facebook remove dups
	public static String removeDuplicates(String s) {
		if (s.length() < 2)
			return s;
		String res = "";
		int[] map = new int[256];
		for (int i = 0; i < s.length(); i++) {
			if (map[s.charAt(i)] == 0) {
				map[s.charAt(i)] = 1;
				res += s.charAt(i);
			}
		}
		return res;
	}

	// find turning number in an array A[i]>A[i-1] &&A[i]>A[i+1]
	public static int TurningNumberIndex(int[] A) {
		if (A.length < 3)
			return -1;
		int i = 0;
		int j = A.length - 1;
		while (i <= j) {// j > i + 1
			int mid = (i + j) / 2;
			if (mid == 0 || mid == A.length - 1)
				return -1;
			if (A[mid] > A[mid - 1] && A[mid] > A[mid + 1])
				return mid;
			if (A[mid] > A[mid - 1] && A[mid] < A[mid + 1])
				i = mid + 1;
			else
				j = mid - 1;
		}
		return -1;
	}

	// Integer Identical to Index
	public static int getNumberSameAsIndex(int[] A) {
		if (A.length == 0)
			return -1;
		int i = 0;
		int j = A.length - 1;
		while (i <= j) {
			int mid = i + (j - i) / 2;
			if (A[mid] == mid)
				return mid;
			else if (A[mid] > mid)
				j = mid - 1;
			else
				i = mid + 1;
		}
		return -1;
	}

	public int findMin(int[] num) {
		// return findMin(num, 0, num.length-1);
		// }

		// public int findMin(int[] A, int low, int high){
		// // if(low==high)
		// // return A[low];
		// // if(low==high-1)
		// // return A[low]<A[high]?A[low]:A[high];
		// if(A[low]<=A[high])
		// return A[low];
		// int mid=(low+high)/2;
		// if(A[mid]>A[high])
		// return findMin(A, mid+1, high);
		// else
		// return findMin(A, low, mid);
		int beg = 0;
		int end = num.length - 1;
		while (beg < end) {
			int mid = (beg + end) / 2;
			if (num[mid] > num[end])
				beg = mid + 1;
			else
				end = mid;
		}
		return num[beg];
	}

	public static int findMinimum(int[] A) {
		int left = 0;
		int right = A.length - 1;

		while (right - left > 1) {
			int mid = (left + right) / 2;

			if (A[mid] <= A[right])
				right = mid - 1;
			else
				left = mid + 1;
		}
		return A[left] > A[right] ? A[right] : A[left];
	}

	public static int findMinimum2(int[] A) {
		if (A.length == 1)
			return A[0];
		int left = 0;
		int right = A.length - 1;
		int mid = left;
		while (A[left] >= A[right]) {
			if (right - left == 1) {
				mid = right;
				break;
			}
			mid = (left + right) / 2;

			if (A[mid] >= A[left])
				left = mid;
			else
				right = mid;
		}
		return A[mid];
	}

	public int findMin2(int[] num) {
		int l = 0;
		int r = num.length - 1;
		if (num[l] < num[r])
			return num[l];

		while (l < r) {
			int m = (l + r) / 2;
			if (num[m] > num[r])
				l = m + 1;
			else
				r = m;
		}
		return num[l];
	}

	// Length of the largest subarray with contiguous elements
	public static int findLength(int[] A) {
		int maxLen = 1;
		for (int i = 0; i < A.length; i++) {
			int min = A[i];
			int max = A[i];
			for (int j = i + 1; j < A.length; j++) {
				max = Math.max(max, A[j]);
				min = Math.min(min, A[j]);
				if (max - min == j - i)
					maxLen = Math.max(maxLen, j - i + 1);
			}
		}
		return maxLen;
	}

	// Print all increasing sequences of length k from first n natural numbers

	public static List<List<Integer>> printSeq(int n, int k) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		printSeqUtil(0, n, k, sol, res, 1);
		return res;
	}

	public static void printSeqUtil(int dep, int n, int k, List<Integer> sol,
			List<List<Integer>> res, int curpos) {
		if (dep == k) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
			return;
		}

		for (int i = curpos; i <= n; i++) {
			sol.add(i);
			printSeqUtil(dep + 1, n, k, sol, res, i + 1);
			sol.remove(sol.size() - 1);
		}
	}

	public static void printLevels(TreeNode root, int low, int high) {
		if (root == null)
			return;
		for (int i = low; i <= high; i++) {
			printLevel(root, i);
			System.out.println();
		}
	}

	public static void printLevel(TreeNode root, int level) {
		if (root == null)
			return;
		if (level == 1)
			System.out.print(root.val + " ");
		printLevel(root.left, level - 1);
		printLevel(root.right, level - 1);
	}

	public static void printLevels2(TreeNode root, int low, int high) {
		if (root == null)
			return;
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		int curlevel = 0;
		int nextlevel = 0;
		int level = 0;
		que.offer(root);
		curlevel++;
		while (!que.isEmpty()) {
			TreeNode top = que.poll();
			curlevel--;
			if (top.left != null) {
				que.offer(top.left);
				nextlevel++;
			}
			if (top.right != null) {
				que.offer(top.right);
				nextlevel++;
			}
			if (level >= low && level <= high)
				System.out.print(top.val);
			if (level > high)
				break;
			if (curlevel == 0) {
				curlevel = nextlevel;
				nextlevel = 0;
				level++;
				System.out.println();
			}
		}
	}

	public String minWindow2(String S, String T) {
		if (S.length() < T.length())
			return "";
		int[] needFind = new int[256];
		for (int i = 0; i < T.length(); i++)
			needFind[T.charAt(i)]++;

		int[] hasFound = new int[256];
		int count = T.length();
		int start = 0;
		int windowStart = 0;
		int windowEnd = 0;
		int minLen = S.length();
		for (int i = 0; i < S.length(); i++) {
			char c = S.charAt(i);
			if (needFind[c] == 0)
				continue;
			hasFound[c]++;
			if (hasFound[c] <= needFind[c])
				count--;
			if (count == 0) {
				if (needFind[S.charAt(start)] == 0
						|| hasFound[S.charAt(start)] > needFind[S.charAt(start)]) {
					if (hasFound[S.charAt(start)] > needFind[S.charAt(start)])
						hasFound[S.charAt(start)]--;
					start++;
				}
				if (i - start + 1 < minLen) {
					windowStart = start;
					windowEnd = i;
					minLen = i - start + 1;
				}
			}
		}
		if (count == 0)
			return S.substring(windowStart, windowEnd + 1);
		return "";
	}

	// Inorder Successor in Binary Search Tree
	public static TreeNode inorderSuccessor(TreeNode root, TreeNode node) {
		if (root == null)
			return null;
		TreeNode successor = null;
		if (node.right != null)
			return leftMostNode(node.right);
		while (root != null) {
			if (root.val > node.val) {
				successor = root;
				root = root.left;
			} else if (root.val < node.val)
				root = root.right;
			else
				break;

		}
		return successor;
	}

	public static TreeNode leftMostNode(TreeNode node) {
		if (node == null)
			return null;
		TreeNode cur = node;
		while (cur.left != null)
			cur = cur.left;
		return cur;
	}

	public static TreeNodeP inorderSucc(TreeNodeP root, TreeNodeP node) {
		if (root == null)
			return null;
		if (node.right != null)
			return leftMostNode(node.right);
		TreeNodeP p = node.parent;
		while (p != null && node == p.right) {
			node = p;
			p = p.parent;
		}
		return p;
	}

	public static TreeNodeP leftMostNode(TreeNodeP node) {
		if (node == null)
			return null;
		TreeNodeP cur = node;
		while (cur.left != null)
			cur = cur.left;
		return cur;
	}

	static class HeapNodeComparator implements Comparator<HeapNode> {

		@Override
		public int compare(HeapNode o1, HeapNode o2) {
			// TODO Auto-generated method stub
			return o1.element - o2.element;
		}

	}

	// Print all elements in sorted order from row and column wise sorted matrix

	public static List<Integer> printSorted(int mat[][]) {
		List<Integer> res = new ArrayList<Integer>();
		int n = mat.length;
		PriorityQueue<HeapNode> heap = new PriorityQueue<HeapNode>(n,
				new HeapNodeComparator());
		for (int i = 0; i < n; i++) {
			HeapNode node = new HeapNode();
			node.element = mat[i][0];
			node.i = i;
			node.j = 1;// Index of next element to be stored from row
			heap.offer(node);
		}

		for (int count = 0; count < n * n; count++) {
			HeapNode node = heap.poll();
			res.add(node.element);

			if (node.j < n) {
				node.element = mat[node.i][node.j];
				node.j++;
				heap.offer(node);
			}
		}
		return res;
	}

	public static TreeNode constructTree(int[] preorder) {
		if (preorder.length == 0)
			return null;
		Stack<TreeNode> stk = new Stack<TreeNode>();
		TreeNode root = new TreeNode(preorder[0]);
		stk.push(root);
		TreeNode temp = null;
		for (int i = 1; i < preorder.length; i++) {
			while (!stk.isEmpty() && preorder[i] > stk.peek().val)
				temp = stk.pop();
			TreeNode node = new TreeNode(preorder[i]);
			if (temp != null) {
				temp.right = node;
			} else {
				stk.peek().left = node;
			}
			stk.push(node);
		}
		return root;
	}

	public static int removeElement(int[] A, int elem) {
		int i = 0;
		int j = 0;
		while (i < A.length) {
			if (A[i] == elem)
				i++;
			else
				A[j++] = A[i++];
		}
		return j;
	}

	public ListNode mergeKLists(ArrayList<ListNode> lists) {
		if (lists == null || lists.isEmpty())
			return null;
		Comparator<ListNode> comp = new Comparator<ListNode>() {

			@Override
			public int compare(ListNode o1, ListNode o2) {
				// TODO Auto-generated method stub
				return o1.val - o2.val;
			}

		};
		PriorityQueue<ListNode> heap = new PriorityQueue<ListNode>(
				lists.size(), comp);
		for (ListNode node : lists) {
			if (node != null)
				heap.add(node);
		}
		ListNode dummy = new ListNode(0);
		ListNode pre = dummy;
		while (!heap.isEmpty()) {
			ListNode node = heap.poll();
			pre.next = node;
			pre = pre.next;
			if (node.next != null)
				heap.add(node.next);
		}
		return dummy.next;
	}

	public List<List<Integer>> subsets2(int[] S) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		Arrays.sort(S);
		subsets2Util(0, S, sol, res);
		return res;
	}

	public void subsets2Util(int cur, int[] S, List<Integer> sol,
			List<List<Integer>> res) {
		res.add(sol);
		if (cur == S.length)
			return;
		for (int i = cur; i < S.length; i++) {
			List<Integer> out = new ArrayList<Integer>(sol);
			out.add(S[i]);
			subsets2Util(i + 1, S, out, res);
		}
	}

	public ListNode reverseBetween2(ListNode head, int m, int n) {
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode cur = head;
		ListNode pre = dummy;
		for (int i = 0; i < m - 1; i++) {
			pre = cur;
			cur = cur.next;
		}

		ListNode p = cur;
		ListNode start = cur;
		cur = cur.next;
		for (int i = 0; i < n - m; i++) {
			ListNode pnext = cur.next;
			cur.next = p;
			p = cur;
			cur = pnext;
		}
		pre.next = p;
		start.next = cur;
		return dummy.next;
	}

	public static int numOfIslands(int[][] matrix) {
		int m = matrix.length;
		int n = matrix[0].length;
		boolean[][] visited = new boolean[m][n];
		int count = 0;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (matrix[i][j] == 1 && !visited[i][j]) {
					int[] area = { 0 };
					dfs(matrix, i, j, visited, area);
					count++;
					System.out.println("size is " + area[0]);
				}
			}
		}
		return count;
	}

	public static void dfs(int[][] matrix, int i, int j, boolean[][] visited,
			int[] area) {
		if (i < 0 || i == matrix.length || j < 0 || j == matrix[0].length
				|| visited[i][j] || matrix[i][j] == 0)
			return;
		visited[i][j] = true;
		area[0]++;
		dfs(matrix, i + 1, j, visited, area);
		dfs(matrix, i - 1, j, visited, area);
		dfs(matrix, i, j + 1, visited, area);
		dfs(matrix, i, j - 1, visited, area);

		// eight directions:diagonal
		dfs(matrix, i + 1, j + 1, visited, area);
		dfs(matrix, i - 1, j - 1, visited, area);
		dfs(matrix, i + 1, j - 1, visited, area);
		dfs(matrix, i - 1, j + 1, visited, area);
	}

	public static String longestCommonString(String s1, String s2) {
		if (s1.isEmpty() || s2.isEmpty())
			return "";
		int[][] dp = new int[s1.length() + 1][s2.length() + 1];
		int max = 0;
		int index = -1;
		for (int i = 1; i <= s1.length(); i++) {
			for (int j = 1; j <= s2.length(); j++) {
				if (s1.charAt(i - 1) == s2.charAt(j - 1)) {
					dp[i][j] = dp[i - 1][j - 1] + 1;
					if (dp[i][j] > max) {
						max = dp[i][j];
						index = i;
					}
				} else
					dp[i][j] = 0;
			}
		}

		return s1.substring(index - max, index);
	}

	public static boolean oneEditApart(String s1, String s2) {
		String small = s1.length() <= s2.length() ? s1 : s2;
		String big = s1.length() <= s2.length() ? s2 : s1;

		int edit = 0;
		if (big.length() - small.length() > 1)
			return false;
		else if (small.length() == big.length()) {
			for (int i = 0; i < small.length(); i++) {
				if (small.charAt(i) != big.charAt(i)) {
					if (++edit > 1)
						return false;
				}
			}
		} else {
			int i = 0;
			while (i < small.length()) {
				if (small.charAt(i) != big.charAt(i + edit)) {
					if (++edit > 1)
						return false;
				} else
					i++;
			}
		}
		return true;
	}

	public static boolean oneEditApart2(String a, String b) {
		String small = a.length() <= b.length() ? a : b;
		String large = a.length() <= b.length() ? b : a;

		int operations = 0;
		if (large.length() - small.length() > 1) {
			return false;
		} else if (large.length() == small.length()) {
			for (int i = 0; i < small.length(); i++)
				if (small.charAt(i) != large.charAt(i) && ++operations > 1)
					return false;
		} else {
			int i = 0;
			while (i < small.length()) {
				if (small.charAt(i) != large.charAt(i + operations)) {
					if (++operations > 1)
						return false;
				} else {
					i++;
				}
			}
		}

		return true;
	}

	public static void intersectionOfTwoBST(TreeNode root1, TreeNode root2) {
		if (root1 == null || root2 == null)
			return;
		if (root1.val == root2.val) {
			System.out.print(root1.val + " ");
			intersectionOfTwoBST(root1.left, root2.left);
			intersectionOfTwoBST(root1.right, root2.right);
		} else if (root1.val > root2.val) {
			intersectionOfTwoBST(root1.left, root2);
			intersectionOfTwoBST(root1, root2.right);
		} else {
			intersectionOfTwoBST(root1, root2.left);
			intersectionOfTwoBST(root1.right, root2);
		}
	}

	public static int areaOfIsland(int[][] matrix, int i, int j) {
		int area = 0;
		Queue<IslandNode> que = new LinkedList<IslandNode>();
		IslandNode node = new IslandNode(i, j);
		que.offer(node);
		boolean[][] visited = new boolean[matrix.length][matrix[0].length];
		visited[i][j] = true;
		while (!que.isEmpty()) {
			IslandNode n = que.poll();
			area++;
			System.out.println("area now is " + area);
			int row = n.row;
			int col = n.col;
			if (col - 1 >= 0 && matrix[row][col - 1] == 1
					&& !visited[row][col - 1]) {
				que.offer(new IslandNode(row, col - 1));
				visited[row][col - 1] = true;
			}
			if (col + 1 < matrix[0].length && matrix[row][col + 1] == 1
					&& !visited[row][col + 1]) {
				que.offer(new IslandNode(row, col + 1));
				visited[row][col + 1] = true;
			}
			if (row - 1 >= 0 && matrix[row - 1][col] == 1
					&& !visited[row - 1][col]) {
				que.offer(new IslandNode(row - 1, col));
				visited[row - 1][col] = true;
			}
			if (row + 1 < matrix.length && matrix[row + 1][col] == 1
					&& !visited[row + 1][col]) {
				que.offer(new IslandNode(row + 1, col));
				visited[row + 1][col] = true;
			}
		}
		return area;
	}

	public static int areaOfIsland2(int[][] matrix, int i, int j) {
		int[] area = { 0 };
		boolean[][] visited = new boolean[matrix.length][matrix[0].length];
		areaOfIsland2(matrix, i, j, visited, area);
		return area[0];
	}

	public static void areaOfIsland2(int[][] matrix, int i, int j,
			boolean[][] visited, int[] area) {
		if (i < 0 || i >= matrix.length || j < 0 || j >= matrix[0].length
				|| visited[i][j])
			return;
		if (!visited[i][j] && matrix[i][j] == 1) {
			System.out.println("row is " + i + ", col is " + j);
			area[0]++;
			visited[i][j] = true;
			areaOfIsland2(matrix, i + 1, j, visited, area);
			areaOfIsland2(matrix, i - 1, j, visited, area);
			areaOfIsland2(matrix, i, j + 1, visited, area);
			areaOfIsland2(matrix, i, j - 1, visited, area);
		}

	}

	public static int findCrossOver(int[] A, int low, int high, int x) {
		if (A[high] <= x)
			return high;
		if (A[low] > x)
			return low;
		int mid = (low + high) / 2;
		if (A[mid] <= x && A[mid + 1] > x)
			return mid;
		if (A[mid] < x)
			return findCrossOver(A, mid + 1, high, x);
		return findCrossOver(A, low, mid - 1, x);
	}

	// This function prints k closest elements to x in arr[].
	// n is the number of elements in arr[]
	public static void findKClosest(int arr[], int x, int k) {
		int left = findCrossOver(arr, 0, arr.length - 1, x);
		int right = left + 1;
		int count = 0;
		if (arr[left] == x)
			left--;
		while (left >= 0 && right < arr.length && count < k) {
			if (x - arr[left] < arr[right] - x)
				System.out.print(arr[left--] + " ");
			else
				System.out.print(arr[right++] + " ");
			count++;
		}
		while (count < k && left >= 0) {
			System.out.println(arr[left--] + " ");
			count++;
		}
		while (count < k && right < arr.length) {
			System.out.println(arr[right++] + " ");
			count++;
		}
	}

	public static int countPalindromeDP1(String s) {
		int[][] dp = new int[s.length()][s.length()];
		for (int i = 0; i < s.length(); i++)
			dp[i][i] = 1;

		for (int i = 0; i < s.length() - 1; i++) {
			if (s.charAt(i) == s.charAt(i + 1))
				dp[i][i + 1] = 1;
		}
		for (int k = 3; k <= s.length(); k++) {
			for (int i = 0; i < s.length() - k + 1; i++) {
				int j = i + k - 1;
				if (dp[i + 1][j - 1] == 1 && s.charAt(i) == s.charAt(j))
					dp[i][j] = 1;
			}
		}
		int count = 0;
		for (int i = 0; i < s.length(); i++) {
			for (int j = 0; j < s.length(); j++) {
				count += dp[i][j];
			}
		}
		return count;
	}

	public static int countPalindromeDP2(String s) {
		int n = s.length();
		int[][] dp = new int[n][n];
		int count = 0;
		for (int i = n - 1; i >= 0; i--) {
			for (int j = i; j < n; j++) {
				if (i == j)
					dp[i][j] = 1;
				else if (j == i + 1) {
					if (s.charAt(i) == s.charAt(j))
						dp[i][j] = 1;
				} else {
					if (dp[i + 1][j - 1] == 1 && s.charAt(i) == s.charAt(j))
						dp[i][j] = 1;
				}
				// if(dp[i][j]==1)
				// System.out.println("start is "+i+", end is "+j);
				count += dp[i][j];
			}
		}
		return count;
	}

	public static int countPalindrome2(String s) {
		int count = s.length();
		int n = s.length();
		for (int i = 0; i < n; i++) {
			int j = i - 1;
			int k = i + 1;
			while (j >= 0 && k < n && s.charAt(j) == s.charAt(k)) {
				count++;
				j--;
				k++;
			}

			j = i;
			k = i + 1;
			while (j >= 0 && k < n && s.charAt(j) == s.charAt(k)) {
				count++;
				j--;
				k++;
			}
		}
		return count;
	}

	// rearragne non-zero elements to the left of the array, zeros to the right
	public static void reArrange(int[] A) {
		int i = 0;
		int j = A.length - 1;
		while (i < j) {
			while (A[i] != 0)
				i++;
			while (A[j] == 0)
				j--;
			if (i < j) {
				int t = A[i];
				A[i] = A[j];
				A[j] = t;
			}
		}
	}

	public static void reArrange2(int[] A) {
		int j = 0;
		for (int i = 0; i < A.length; i++) {
			if (A[i] != 0)
				A[j++] = A[i];
		}
		while (j < A.length) {
			A[j++] = 0;
		}
	}

	// iterative
	// Power set P(S) of a set S is the set of all subsets of S.
	public static List<List<Integer>> powerSet(int[] s) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		int n = s.length;
		int setSize = (int) Math.pow(2, n);
		for (int i = 0; i < setSize; i++) {
			List<Integer> sol = new ArrayList<Integer>();
			for (int j = 0; j < n; j++) {
				if ((i & 1 << j) != 0)
					sol.add(s[j]);
			}
			res.add(sol);
		}
		return res;
	}

	// recursive
	public static List<List<Integer>> powerSet2(int[] s) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		powerSetUtil(0, s, sol, res);
		return res;
	}

	public static void powerSetUtil(int cur, int[] s, List<Integer> sol,
			List<List<Integer>> res) {
		res.add(sol);
		if (cur == s.length)
			return;
		for (int i = cur; i < s.length; i++) {
			List<Integer> out = new ArrayList<Integer>(sol);
			out.add(s[i]);
			powerSetUtil(i + 1, s, out, res);
		}
	}

	// 给一个矩阵A，要求输出一个矩阵B，要求B[i,j]=sum(A[l,k]) l<=i,k<=j;

	public static int[][] transformMatrix(int[][] A) {
		int m = A.length;
		int n = A[0].length;
		int[][] B = new int[m][n];
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				int sum = 0;
				for (int k = i; k >= 0; k--) {
					for (int l = j; l >= 0; l--) {
						sum += A[k][l];
					}
				}
				B[i][j] = sum;
			}
		}
		return B;
	}

	// optimize the solution:Summed area table
	public static int[][] transformMatrix2(int[][] A) {
		int m = A.length;
		int n = A[0].length;
		int[][] B = new int[m][n];
		B[0][0] = A[0][0];
		for (int i = 1; i < m; i++) {
			B[i][0] = B[i - 1][0] + A[i][0];
		}
		for (int i = 1; i < n; i++)
			B[0][i] = B[0][i - 1] + A[0][i];

		for (int i = 1; i < m; i++) {
			for (int j = 1; j < n; j++) {
				B[i][j] = B[i - 1][j] + B[i][j - 1] - B[i - 1][j - 1] + A[i][j];
			}
		}
		return B;
	}

	public static TreeNode findKthTreeNode(TreeNode root, int k) {
		if (root == null)
			return null;
		int leftcounts = countNodes(root.left);
		if (leftcounts == k - 1)
			return root;
		else if (leftcounts > k - 1)
			return findKthTreeNode(root.left, k);
		else
			return findKthTreeNode(root.right, k - leftcounts - 1);

	}

	public static ListNode interleave(ListNode p, ListNode q) {
		if (p == null || q == null)
			return p == null ? q : p;
		// ListNode dummy=new ListNode(0);
		// ListNode pre=dummy;
		ListNode head = p;
		while (p != null && q != null) {
			ListNode pnext = p.next;
			p.next = q;
			ListNode qnext = q.next;
			q.next = pnext;

			p = pnext;
			q = qnext;
		}
		return head;
	}

	public static ListNode interleave2(ListNode p, ListNode q) {
		if (p == null || q == null)
			return p == null ? q : p;
		ListNode pnext = p.next;
		ListNode qnext = q.next;
		p.next = q;
		q.next = pnext;
		pnext = interleave2(pnext, qnext);

		return p;

	}

	public static int longestConsecutive2(int[] num) {
		HashMap<Integer, Boolean> map = new HashMap<Integer, Boolean>();
		for (int i : num)
			map.put(i, false);
		int max = 1;
		for (int i = 0; i < num.length; i++) {
			if (map.get(num[i]))
				continue;
			int t = num[i];
			int cur_max = 1;
			while (map.containsKey(t - 1)) {
				t--;
				cur_max++;
				map.put(t, true);
			}
			t = num[i];
			while (map.containsKey(t + 1)) {
				t++;
				cur_max++;
				map.put(t, true);
			}
			max = Math.max(max, cur_max);
		}
		return max;
	}

	public static void levelAverage(TreeNode root) {
		if (root == null)
			return;
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		int curlevel = 0;
		int nextlevel = 0;
		que.add(root);
		curlevel++;
		int sum = 0;
		int count = 0;
		int level = 0;
		while (!que.isEmpty()) {
			TreeNode top = que.poll();
			curlevel--;
			sum += top.val;
			count++;
			if (top.left != null) {
				que.offer(top.left);
				nextlevel++;
			}
			if (top.right != null) {
				que.offer(top.right);
				nextlevel++;
			}
			if (curlevel == 0) {
				System.out.println("level " + level + " average is " + sum
						/ count);
				level++;
				curlevel = nextlevel;
				nextlevel = 0;
				sum = 0;
				count = 0;
			}
		}
	}

	public static TreeNode findLCA(TreeNode root, TreeNode s, TreeNode b) {
		if (root == null)
			return null;
		TreeNode p = root;
		while (p.val < s.val || p.val > b.val) {
			while (p.val < s.val)
				p = p.right;
			while (p.val > b.val)
				p = p.left;
		}
		// p.getData() >= s.getData() && p.getData() <= b.getData().
		return p;
	}

	public static List<Integer> printTopViewOfBT(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		if (root == null)
			return res;
		HashSet<Integer> set = new HashSet<Integer>();
		Queue<TopViewNode> que = new LinkedList<TopViewNode>();
		que.add(new TopViewNode(root, 0));
		while (!que.isEmpty()) {
			TopViewNode topNode = que.remove();
			TreeNode node = topNode.node;
			int hd = topNode.hd;
			if (!set.contains(hd)) {
				res.add(node.val);
				set.add(hd);
			}
			if (node.left != null)
				que.add(new TopViewNode(node.left, hd - 1));
			if (node.right != null)
				que.add(new TopViewNode(node.right, hd + 1));

		}
		return res;
	}

	// Length of Longest Arithmetic Progression //the given set is sorted
	public static int lengthLongestAP(int[] A) {
		int n = A.length;
		// dp[i][j] in this table stores LLAP with A[i] and A[j] as first two
		// elements of AP and j > i.
		int[][] dp = new int[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				dp[i][j] = 2;
		}
		int longest = 2;
		for (int j = n - 2; j >= 0; j--) {
			int i = j - 1;
			int k = i + 1;
			while (i >= 0 && k < n) {
				if (A[i] + A[k] > 2 * A[j])
					i--;
				else if (A[i] + A[k] < 2 * A[j])
					k++;
				else {
					dp[i][j] = dp[j][k] + 1;
					longest = Math.max(longest, dp[i][j]);
					i--;
					k++;
				}
			}
		}
		return longest;
	}

	public static boolean detectLoop(TreeNode root) {
		HashSet<TreeNode> visited = new HashSet<TreeNode>();
		return detectLoop(root, visited);
	}

	public static boolean detectLoop(TreeNode root, HashSet<TreeNode> set) {
		if (root == null)
			return false;
		if (set.contains(root))
			return true;
		set.add(root);
		return detectLoop(root.left, set) || detectLoop(root.right, set);
	}

	// Given an array of jobs where every job has a deadline and associated
	// profit if the job is finished before the deadline.
	// It is also given that every job takes single unit of time, so the minimum
	// possible deadline for any job is 1.
	// How to maximize total profit if only one job can be scheduled at a time.

	public static void jobScheduling(Job[] jobs) {
		Comparator<Job> comp = new Comparator<Job>() {

			@Override
			public int compare(Job o1, Job o2) {
				// TODO Auto-generated method stub
				return o2.profit - o1.profit;
			}

		};
		Arrays.sort(jobs, comp);
		int n = jobs.length;
		boolean[] slots = new boolean[n];
		int[] res = new int[n];

		for (int i = 0; i < jobs.length; i++) {
			for (int j = Math.min(n, jobs[i].dead) - 1; j >= 0; j--) {// 0 based
				if (!slots[j]) {
					slots[j] = true;
					res[j] = i;
					break;
				}
			}
		}

		for (int i = 0; i < res.length; i++) {
			if (slots[i])
				System.out.print(jobs[res[i]].id + " ");
		}
		System.out.println();
	}

	// Minimum Number of Platforms Required for a Railway/Bus Station

	public int findPlatform(int arr[], int dep[]) {
		Arrays.sort(arr);
		Arrays.sort(dep);

		int max = 0;
		int platforms = 1;
		int i = 1;
		int j = 0;
		while (i < arr.length && j < dep.length) {
			if (arr[i] < dep[j]) {
				platforms++;
				max = max < platforms ? platforms : max;
				i++;
			} else {
				platforms--;
				j++;
			}
		}
		return max;
	}

	// Multiply two polynomials
	// A[] represents coefficients of first polynomial
	// B[] represents coefficients of second polynomial
	// m and n are sizes of A[] and B[] respectively
	public int[] multiply(int A[], int B[]) {
		int m = A.length;
		int n = B.length;
		int[] poly = new int[m + n - 1];

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				poly[i + j] += A[i] * B[j];
			}
		}
		return poly;
	}

	/**
	 * Write a function that determines whether a array contains duplicate
	 * characters within k indices of each other
	 */

	public static boolean duplicateWithinkIndices(int[] A, int k) {
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < A.length; i++) {
			if (!map.containsKey(A[i]))
				map.put(A[i], i);
			else {
				int idx = map.get(A[i]);
				if (i - idx <= k)
					return true;
				else
					map.put(A[i], i);
			}
		}
		return false;
	}

	public TreeNode UpsideDownBinaryTree(TreeNode root) {
		TreeNode p = root;
		TreeNode parent = null;
		while (p != null) {
			TreeNode left = p.left;
			left.left = p;
			left.right = p.right;
			p.left = null;
			p.right = null;
			parent = p;
			p = left;
		}
		return parent;
	}

	// make |sumA-sumB| minimized by swap elements between A and B

	public static int adjust(int[] A, int[] B) {

		int sumA = 0;
		int sumB = 0;
		for (int i = 0; i < A.length; i++) {
			if ((A[i] - B[i]) * (sumA - sumB) > 0) {
				int t = A[i];
				A[i] = B[i];
				B[i] = t;
			}
			sumA += A[i];
			sumB += B[i];
		}
		return Math.abs(sumA - sumB);
	}

	public static List<Integer> maxSlidingWindow(int A[], int w) {
		List<Integer> res = new ArrayList<Integer>();
		Deque<Integer> que = new ArrayDeque<Integer>();
		for (int i = 0; i < w; i++) {
			while (!que.isEmpty() && A[i] >= A[que.peekLast()])
				que.pollLast();
			que.offerLast(i);
		}

		for (int i = w; i < A.length; i++) {
			res.add(A[que.peekFirst()]);
			while (!que.isEmpty() && A[i] >= A[que.peekLast()])
				que.pollLast();
			while (!que.isEmpty() && que.peekFirst() <= i - w)
				que.pollFirst();
			que.offerLast(i);
		}
		res.add(que.pollFirst());
		return res;
	}

	public static int[] minSlidingWindow(int[] A, int w) {
		int n = A.length;
		int[] B = new int[n - w + 1];
		int[] queue = new int[n];
		int head = 0, tail = 0;
		for (int i = 0; i < n; i++) {
			while (head < tail && A[i] < A[queue[tail - 1]])
				tail--;
			queue[tail++] = i;

			if (i >= w - 1) {
				while (head < tail && i - queue[head] >= w)
					head++;
				B[i - w + 1] = A[queue[head]];
			}
		}
		return B;
	}

	public static int[] minSlidingWindow2(int[] A, int w) {
		int n = A.length;
		int[] B = new int[n - w + 1];
		Deque<Integer> que = new ArrayDeque<Integer>();

		for (int i = 0; i < w; i++) {
			while (!que.isEmpty() && A[i] < A[que.peekLast()])
				que.pollLast();
			que.offerLast(i);
		}

		for (int i = w; i < n; i++) {
			B[i - w] = A[que.peekFirst()];
			while (!que.isEmpty() && A[i] < que.peekLast())
				que.pollLast();

			while (!que.isEmpty() && i - que.peekFirst() >= w)
				que.pollFirst();
			que.offerLast(i);
		}
		B[n - w] = A[que.pollFirst()];
		return B;
	}

	// 大意是有个MXN的棋盘，然后每个格子上有一个价值，需要从左上角走到右下角再走回来，但每个格子只能走一次，请问来回的最大收获是多少
	// 题目链接： http://soj.me/1767
	// 考查从 （1,1）到（m,n）找两条不相交的路径使得它们的权值和最大。
	//
	// 法1：动态规划
	//
	// 设 f [ k ][ i ][ j ] 表示第 k 步，第 1 条路径走到第 i 行，第 2 条路径走到第 j 行的最大权值和。
	// 状态转移方程：
	// f [ k ][ i ][ j ] = max { f [ k - 1 ][ i - 1 ][ j ], f [ k - 1 ][ i ][ j
	// - 1 ], f [ k - 1 ][ i ][ j ], f [ k - 1 ][ i - 1 ][ j - 1 ] } + map[ i ][
	// k - i ] + map[ j ][ k - j ]
	// ( 2 <= k <= m + n, 1 <= i, j <= min { m, k - 1 }, i != j )

	public int maxRoundTripBenifit(int[][] map) {
		int m = map.length;
		int n = map[0].length;
		int[][][] dp = new int[m + n + 1][m + 1][m + 1];
		for (int k = 2; k <= m + n; k++) {
			for (int i = 1; i <= m && i <= k - 1; i++) {
				for (int j = 1; j <= m && j <= k - 1; j++) {
					if (k != m + n && i == j)
						continue;
					dp[k][i][j] = Math
							.max(dp[k - 1][i - 1][j], Math.max(
									dp[k - 1][i][j - 1], Math.max(
											dp[k - 1][i - 1][j - 1],
											dp[k - 1][i][j])))
							+ map[i][k - i] + map[j][k - j];
				}
			}
		}
		return dp[m + n][m][m];
	}

	// coins in line

	// F(i, j) represents the maximum value the user can collect from i'th coin
	// to j'th coin.
	//
	// F(i, j) = Max(Vi + min(F(i+2, j), F(i+1, j-1) ),
	// Vj + min(F(i+1, j-1), F(i, j-2) ))
	// Base Cases
	// F(i, j) = Vi If j == i
	// F(i, j) = max(Vi, Vj) If j == i+1
	public static int coins(int[] A) {
		int n = A.length;
		int[][] dp = new int[n][n];

		for (int gap = 0; gap < n; gap++) {
			for (int i = 0; i < n - gap; i++) {
				int j = i + gap;
				int x = i + 2 <= j ? dp[i + 2][j] : 0;
				int y = i + 1 <= j - 1 ? dp[i + 1][j - 1] : 0;
				int z = i <= j - 2 ? dp[i][j - 2] : 0;

				dp[i][j] = Math.max(A[i] + Math.min(x, y),
						A[j] + Math.min(y, z));
			}
		}
		return dp[0][n - 1];
	}

	// Given two strings containing digits, return the one which represents the
	// largest integer once the digits have been sorted in non-increasing order.
	// “245” -> 542
	// “178” -> 871
	// return 178

	public static String maxString(String s1, String s2) {
		if (s1.length() != s2.length())
			return s1.length() < s2.length() ? s2 : s1;
		// char[] ch1=s1.toCharArray();
		// Arrays.sort(ch1);
		// char[] ch2=s2.toCharArray();
		// Arrays.sort(ch2);
		//
		// for(int i=ch1.length-1;i>=0;i--){
		// if(ch1[i]>ch2[i])
		// return s1;
		// else if(ch1[i]<ch2[i])
		// return s2;
		// }
		// return s1;
		int[] ch1 = new int[10];
		int[] ch2 = new int[10];
		for (int i = 0; i < s1.length(); i++) {
			ch1[s1.charAt(i) - '0']++;
			ch2[s2.charAt(i) - '0']++;
		}
		for (int i = 9; i >= 0; i--) {
			if (ch1[i] > ch2[i])
				return s1;
			else if (ch1[i] < ch2[i])
				return s2;
		}
		return s1;
	}

	public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
		if (headA == null || headB == null)
			return null;
		ListNode node1 = headA;
		ListNode node2 = headB;

		while (node1 != node2) {
			node1 = node1.next;
			node2 = node2.next;
			if (node1 == node2)
				return node1;
			if (node1 == null)
				node1 = headB;
			if (node2 == null)
				node2 = headA;
		}
		return node1;
	}

	public int atoi2(String str) {
		str = str.trim();
		if (str.isEmpty())
			return 0;
		boolean neg = false;
		boolean overflow = false;
		int i = 0;
		if (str.charAt(i) == '-') {
			neg = true;
			i++;
		} else if (str.charAt(i) == '+')
			i++;
		int res = 0;
		while (i < str.length()) {
			int num = str.charAt(i) - '0';
			if (num >= 0 && num <= 9) {
				if ((Integer.MAX_VALUE - num) / 10 < res) {
					overflow = true;
					break;
				}
				res = res * 10 + num;
			} else
				break;
			i++;
		}

		if (neg) {
			if (overflow)
				return Integer.MIN_VALUE;
			return -res;
		} else {
			if (overflow)
				return Integer.MAX_VALUE;
			return res;
		}

	}

	public double findMedianSortedArrays2(int A[], int B[]) {
		int m = A.length;
		int n = B.length;

		if ((m + n) % 2 == 0)
			return (findKth2(A, 0, m, B, 0, n, (m + n) / 2) + findKth2(A, 0, m,
					B, 0, n, (m + n) / 2 + 1)) / 2.0;
		return findKth2(A, 0, m, B, 0, n, (m + n) / 2 + 1);
	}

	public double findKth2(int[] A, int aoffset, int m, int[] B, int boffset,
			int n, int k) {
		if (m > n)
			return findKth2(B, boffset, n, A, aoffset, m, k);
		if (m == 0)
			return B[k - 1];
		if (k == 1)
			return Math.min(A[aoffset], B[boffset]);
		int pa = Math.min(m, k / 2);
		int pb = k - pa;

		if (A[aoffset + pa - 1] >= B[boffset + pb - 1])
			return findKth2(A, aoffset, m, B, boffset + pb, n - pb, k - pb);
		else
			return findKth2(A, aoffset + pa, m - pa, B, boffset, n, k - pa);
	}

	public static String longestPalindromeDP(String s) {
		int max = 1;
		int start = 1;
		int end = 1;
		boolean[][] dp = new boolean[s.length()][s.length()];

		for (int gap = 0; gap < s.length(); gap++) {
			for (int i = 0; i < s.length() - gap; i++) {
				int j = i + gap;
				if (i == j)
					dp[i][j] = true;
				else if (i + 1 == j)
					dp[i][j] = s.charAt(i) == s.charAt(j);
				else
					dp[i][j] = s.charAt(i) == s.charAt(j) && dp[i + 1][j - 1];
				if (dp[i][j]) {
					if (gap + 1 > max) {
						max = gap + 1;
						start = i;
						end = j;
					}
				}
			}
		}
		return s.substring(start, end + 1);
	}

	public static int findPeakElement(int[] num) {
		int i = 0;
		int j = num.length;
		while (i < j) {
			int mid = (i + j) / 2;
			if ((mid == 0 || num[mid] > num[mid - 1])
					&& (mid == num.length - 1 || num[mid] > num[mid + 1]))
				return mid;
			else if (mid > 0 && num[mid - 1] > num[mid])
				j = mid - 1;
			else
				i = mid + 1;
		}
		return i;
	}

	// 给一个int数组，和一个数X，求K，使得A[0...K-1]里与X相等个数与A[K...N]不相等的
	// 个数相等， 要求输入int[] A,int XK，K不存在时返回-1
	// 要求时间复杂度为O(N),空间复杂度为O(1)
	public static int findIndex(int[] A, int x) {
		int numNotX = 0;
		for (int i = 0; i < A.length; i++) {
			if (A[i] != x)
				numNotX++;
		}

		int curX = 0;
		int curNotX = 0;
		for (int i = 0; i < A.length; i++) {
			if (A[i] == x)
				curX++;
			else
				curNotX++;
			if (curX == numNotX - curNotX)
				return i + 1;
		}
		return -1;
	}

	// Find two numbers from BST which sum to given number K

	public static void twoSumOfBST(TreeNode root, int target) {
		if (root == null)
			return;
		TreeNode node1 = root, node2 = root;
		Stack<TreeNode> stk1 = new Stack<TreeNode>();
		Stack<TreeNode> stk2 = new Stack<TreeNode>();
		while (node1.left != null) {
			stk1.push(node1);
			node1 = node1.left;
		}
		while (node2.right != null) {
			stk2.push(node2);
			node2 = node2.right;
		}

		while (node1.val < node2.val) {
			int sum = node1.val + node2.val;
			if (sum == target) {
				System.out.println(node1.val + " " + node2.val);
				node1 = stk1.pop();
				node2 = stk2.pop();
				System.out.println("left pop " + node1.val);
				System.out.println("right pop " + node2.val);

			} else if (sum > target) {
				if (node2.left != null) {
					node2 = node2.left;
					while (node2.right != null) {
						stk2.push(node2);
						node2 = node2.right;
					}
				} else {
					node2 = stk2.pop();
					System.out.println("right pop " + node2.val);
				}
			} else {
				if (node1.right != null) {
					node1 = node1.right;
					while (node1.left != null) {
						stk1.push(node1);
						node1 = node1.left;
					}
				} else {
					node1 = stk1.pop();
					System.out.println("left pop " + node1.val);
				}
			}
			System.out.println("left=" + node1.val);
			System.out.println("right=" + node2.val);
		}
	}

	public ListNode reverseKGroup2(ListNode head, int k) {
		if (head == null || head.next == null)
			return head;
		ListNode dummy = new ListNode(0);
		dummy.next = head;

		ListNode pre = dummy;
		ListNode cur = head;
		int i = 0;
		while (cur != null) {
			i++;
			if (i % k == 0) {
				pre = reverseList2(pre, cur.next);
				cur = pre.next;
			} else
				cur = cur.next;
		}
		return dummy.next;
	}

	public ListNode reverseList2(ListNode pre, ListNode next) {
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

	public static int numDecodingsConst(String s) {
		if (s.length() == 0 || s.charAt(0) == '0')
			return 0;
		if (s.length() == 1)
			return 1;
		int res = 0;
		int first = 1;
		int second = 1;

		for (int i = 2; i <= s.length(); i++) {
			char c1 = s.charAt(i - 1);
			char c2 = s.charAt(i - 2);
			if (c1 != '0')
				res = first;
			if (c2 == '1' || c2 == '2' && c1 < '7')
				res += second;

			second = first;
			first = res;
		}
		return res;
	}

	public static String minWindow3(String S, String T) {
		if (S.length() < T.length())
			return "";
		int[] needToFind = new int[256];
		for (int i = 0; i < T.length(); i++)
			needToFind[T.charAt(i)]++;
		int count = T.length();
		int[] hasFound = new int[256];

		int start = 0;
		int min = S.length() + 1;

		int winStart = 0;
		int winEnd = 0;

		for (int i = 0; i < S.length(); i++) {
			char c = S.charAt(i);
			// if(needToFind[c]==0)
			// continue;
			hasFound[c]++;
			if (hasFound[c] <= needToFind[c])
				count--;
			if (count == 0) {
				while (hasFound[S.charAt(start)] > needToFind[S.charAt(start)]) {
					// if(hasFound[S.charAt(start)]>needToFind[S.charAt(start)])
					hasFound[S.charAt(start)]--;
					start++;
				}
				if (i - start + 1 < min) {
					min = i - start + 1;
					winStart = start;
					winEnd = i;
				}
			}
		}
		if (count == 0)
			return S.substring(winStart, winEnd + 1);
		return "";
	}

	public UndirectedGraphNode cloneGraph2(UndirectedGraphNode node) {
		if (node == null)
			return null;
		UndirectedGraphNode copy = new UndirectedGraphNode(node.label);
		Map<UndirectedGraphNode, UndirectedGraphNode> map = new HashMap<UndirectedGraphNode, UndirectedGraphNode>();
		map.put(node, copy);

		Queue<UndirectedGraphNode> que = new LinkedList<UndirectedGraphNode>();
		que.add(node);
		while (!que.isEmpty()) {
			UndirectedGraphNode top = que.remove();
			List<UndirectedGraphNode> neighbors = top.neighbors;
			for (int i = 0; i < neighbors.size(); i++) {
				UndirectedGraphNode nb = neighbors.get(i);
				if (!map.containsKey(nb)) {
					UndirectedGraphNode copynode = new UndirectedGraphNode(
							nb.label);
					que.add(nb);
					map.put(nb, copynode);
					map.get(top).neighbors.add(copynode);
				} else
					map.get(top).neighbors.add(map.get(nb));

			}
		}
		return copy;
	}

	public static int candy(int[] ratings) {
		int[] candy = new int[ratings.length];
		Arrays.fill(candy, 1);
		for (int i = 1; i < ratings.length; i++) {
			if (ratings[i] > ratings[i - 1])
				candy[i] = candy[i - 1] + 1;
		}

		System.out.println(Arrays.toString(candy));
		for (int i = ratings.length - 2; i >= 0; i--) {
			if (ratings[i] > ratings[i + 1])
				candy[i] = Math.max(candy[i], candy[i + 1] + 1);
		}
		int candies = 0;
		for (int i = 0; i < candy.length; i++)
			candies += candy[i];

		System.out.println(Arrays.toString(candy));
		return candies;
	}

	public boolean isScramble(String s1, String s2) {
		if (s1.length() != s2.length())
			return false;
		int len = s1.length();
		// dp[i][j][k] means substrings that s1 from i, s2 from j, length =k are
		// scramble
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

	public boolean isScrambleRecur(String s1, String s2) {
		if (s1.length() != s2.length())
			return false;
		if (s1.equals(s2))
			return true;

		for (int i = 1; i < s1.length(); i++) {
			String s11 = s1.substring(0, i);
			String s12 = s1.substring(i);

			String s21 = s2.substring(0, i);
			String s22 = s2.substring(i);

			if (isScramble(s11, s21) && isScramble(s12, s22))
				return true;
			s21 = s2.substring(s2.length() - i);
			s22 = s2.substring(0, s2.length() - i);

			if (isScramble(s11, s22) && isScramble(s12, s21))
				return true;
		}
		return false;
	}

	public static int visibleNodes(TreeNode root) {
		if (root == null)
			return 0;
		int[] left = { 0 };
		int[] right = { 0 };
		visibleNodes(root.left, root.val, left);
		visibleNodes(root.right, root.val, right);
		return left[0] + right[0] + 1;
	}

	public static void visibleNodes(TreeNode root, int max, int[] count) {
		if (root == null)
			return;
		if (root.val > max) {
			max = root.val;
			count[0]++;
			System.out.print(max + " ");
		}
		visibleNodes(root.left, max, count);
		visibleNodes(root.right, max, count);
	}

	public int compareVersion(String version1, String version2) {
		String[] str1 = version1.split("\\.");
		String[] str2 = version2.split("\\.");
		int i = 0;
		for (; i < str1.length && i < str2.length; i++) {
			if (toNum(str1[i]) > toNum(str2[i]))
				return 1;
			else if (toNum(str1[i]) < toNum(str2[i]))
				return -1;
		}
		while (i < str1.length) {
			if (toNum(str1[i++]) != 0)
				return 1;
		}
		while (i < str2.length) {
			if (toNum(str2[i++]) != 0)
				return -1;
		}
		return 0;
	}

	public int toNum(String s) {
		return Integer.parseInt(s);
	}

	public String fractionToDecimal(int numerator, int denominator) {
		if (numerator == 0)
			return "0";
		boolean sign = (numerator > 0 && denominator < 0)
				|| (numerator < 0 && denominator > 0);
		long num = Math.abs((long) numerator);
		long denom = Math.abs((long) denominator);

		StringBuilder sb = new StringBuilder();
		if (sign)
			sb.append("-");
		sb.append(num / denom);
		if (num % denom == 0)
			return sb.toString();
		else
			sb.append(".");
		HashMap<Long, Integer> map = new HashMap<Long, Integer>();
		long rem = num % denom;
		while (rem > 0) {
			if (map.containsKey(rem)) {
				sb.insert(map.get(rem), "(");
				sb.append(")");
				break;
			} else
				map.put(rem, sb.length());
			rem *= 10;
			sb.append(rem / denom);
			rem %= denom;
		}

		Iterator<Long> it = map.keySet().iterator();
		while (it.hasNext()) {
			Long key = it.next();
			System.out.println(key + " , " + map.get(key));
		}
		return sb.toString();
	}

	// rotated array duplicates
	public boolean search2(int[] A, int target) {
		int beg = 0;
		int end = A.length - 1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A[mid] == target)
				return true;
			if (A[beg] < A[mid]) {
				if (A[beg] <= target && target < A[mid])
					end = mid - 1;
				else
					beg = mid + 1;
			} else if (A[mid] < A[beg]) {
				if (A[mid] < target && target <= A[end])
					beg = mid + 1;
				else
					end = mid - 1;
			} else
				beg++;
		}
		return false;
	}

	public int knapsack(int W, int[] vals, int[] wt) {
		int n = vals.length;
		int[][] dp = new int[n + 1][W + 1];

		for (int i = 0; i <= n; i++) {
			for (int w = 0; w <= W; w++) {
				if (i == 0 || w == 0)
					dp[i][w] = 0;
				else if (w >= wt[i - 1]) {
					dp[i][w] = Math.max(vals[i - 1] + dp[i - 1][w - wt[i - 1]],
							dp[i - 1][w]);
				} else
					dp[i][w] = dp[i - 1][w];
			}
		}
		return dp[n][W];
	}

	public int knapsackReduceSpace(int W, int[] vals, int[] wt) {
		int n = vals.length;
		int[] dp = new int[W + 1];

		for (int i = 1; i <= n; i++) {
			for (int w = W; w >= 0; w--) {
				if (w >= wt[i - 1])
					dp[w] = Math.max(vals[i - 1] + dp[w - wt[i - 1]], dp[w]);
			}
		}
		return dp[W];
	}

	public int trap2(int[] A) {
		int n = A.length;
		if (n < 2)
			return 0;
		int[] left = new int[n];
		int[] right = new int[n];

		left[0] = 0;
		int max = A[0];
		for (int i = 1; i < n; i++) {
			max = Math.max(max, A[i]);
			left[i] = max;
		}
		right[n - 1] = 0;
		max = A[n - 1];
		for (int i = n - 2; i >= 0; i--) {
			max = Math.max(max, A[i]);
			right[i] = max;
		}
		System.out.println(Arrays.toString(left));
		System.out.println(Arrays.toString(right));
		int total = 0;
		for (int i = 0; i < n; i++) {
			int trap = Math.min(left[i], right[i]) - A[i];
			if (trap > 0)
				total += trap;
		}
		return total;
	}

	public int trap3(int[] A) {
		if (A == null || A.length == 0)
			return 0;
		int max = 0;
		int res = 0;
		int[] container = new int[A.length];
		for (int i = 0; i < A.length; i++) {
			container[i] = max;
			max = Math.max(max, A[i]);
		}

		System.out.println(Arrays.toString(container));
		max = 0;
		for (int i = A.length - 1; i >= 0; i--) {
			container[i] = Math.min(max, container[i]);
			max = Math.max(max, A[i]);
			res += container[i] - A[i] > 0 ? container[i] - A[i] : 0;
		}
		return res;
	}

	// convert to excel
	public String convertToTitle(int n) {
		String res = "";
		while (n > 0) {
			n--;
			char c = (char) (n % 26 + 'A');
			res = c + res;
			n /= 26;
		}
		return res;
	}

	public void flatten3(TreeNode root) {
		if (root == null || root.left == null && root.right == null)
			return;
		if (root.left != null) {
			TreeNode right = root.right;
			root.right = root.left;
			TreeNode node = root.left;
			while (node.right != null)
				node = node.right;
			node.right = right;
			root.left = null;
		}
		flatten(root.right);
	}

	public int canCompleteCircuit2(int[] gas, int[] cost) {
		int total = 0;
		int sum = 0;
		int start = 0;
		for (int i = 0; i < gas.length; i++) {
			sum += gas[i] - cost[i];
			total += gas[i] - cost[i];
			if (sum < 0) {
				start = i + 1;
				sum = 0;
			}
		}
		return total >= 0 ? start : -1;
	}

	public int calculateMinimumHP(int[][] dungeon) {
		int m = dungeon.length;
		int n = dungeon[0].length;

		int[][] dp = new int[m][n];
		dp[m - 1][n - 1] = Math.max(0 - dungeon[m - 1][n - 1], 0);

		for (int i = m - 2; i >= 0; i--) {
			dp[i][n - 1] = Math.max(dp[i + 1][n - 1] - dungeon[i][n - 1], 0);
		}

		for (int j = n - 2; j >= 0; j--) {
			dp[m - 1][j] = Math.max(dp[m - 1][j + 1] - dungeon[m - 1][j], 0);
		}

		for (int i = m - 2; i >= 0; i--) {
			for (int j = n - 2; j >= 0; j--) {
				dp[i][j] = Math.max(Math.min(dp[i + 1][j], dp[i][j + 1])
						- dungeon[i][j], 0);
			}
		}
		return dp[0][0] + 1;
	}

	public String largestNumber(int[] num) {
		StringBuilder sb = new StringBuilder();
		String[] nums = new String[num.length];
		for (int i = 0; i < num.length; i++)
			nums[i] = "" + num[i];
		Arrays.sort(nums, new Comparator<String>() {

			@Override
			public int compare(String o1, String o2) {
				// TODO Auto-generated method stub
				String s1 = o1 + o2;
				String s2 = o2 + o1;
				return s2.compareTo(s1);
			}

		});
		;
		if (nums[0].equals("0"))
			return "0";
		for (int i = 0; i < num.length; i++)
			sb.append(nums[i]);

		return sb.toString();
	}

	public double pow(double x, int n) {
		if (n == 0)
			return 1;
		boolean neg = false;
		if (n < 0) {
			neg = true;
			n = -n;
		}
		double res = pow(x, n / 2);
		if (n % 2 == 0)
			res *= res;
		else
			res *= res * x;
		System.out.println(neg + " " + res);
		if (neg)
			return 1 / res;
		return res;
	}

	// Fill two instances of all numbers from 1 to n in a specific way
	// Given a number n, create an array of size 2n
	// such that the array contains 2 instances of every number from 1 to n,
	// and the number of elements between two instances of a number i is equal
	// to i.
	// If such a configuration is not possible, then print the same.

	public void fill(int n) {
		int[] res = new int[2 * n];

		if (fillUtil(res, n)) {
			System.out.println(Arrays.toString(res));
		} else
			System.out.println("not possible");
	}

	public boolean fillUtil(int[] res, int cur) {
		if (cur == 0)
			return true;
		for (int i = 0; i < res.length - cur - 1; i++) {
			if (res[i] == 0 && res[i + cur + 1] == 0) {
				res[i] = res[i + cur + 1] = cur;
				if (fillUtil(res, cur - 1))
					return true;
				res[i] = res[i + cur + 1] = 0;
			}
		}
		return false;
	}

	public List<Interval> merge2(List<Interval> intervals) {
		List<Interval> res = new ArrayList<Interval>();
		if (intervals.size() < 2)
			return intervals;
		Comparator<Interval> cp = new Comparator<Interval>() {
			@Override
			public int compare(Interval i1, Interval i2) {
				return i1.start - i2.start;
			}
		};
		Collections.sort(intervals, cp);
		res.add(intervals.get(0));
		for (int i = 1; i < intervals.size(); i++) {
			Interval last = res.get(res.size() - 1);
			Interval interval = intervals.get(i);
			if (interval.start > last.end)
				res.add(interval);
			else
				last.end = Math.max(interval.end, last.end);
		}
		return res;
	}

	public ListNode rotateRight2(ListNode head, int n) {
		if (head == null)
			return head;
		int len = 0;
		ListNode cur = head;
		while (cur != null) {
			len++;
			cur = cur.next;
		}
		n = n % len;
		if (n == 0)
			return head;
		ListNode fast = head;
		ListNode slow = head;

		for (int i = 0; i < n; i++) {
			fast = fast.next;
		}
		while (fast.next != null) {
			fast = fast.next;
			slow = slow.next;
		}
		ListNode node = slow.next;
		slow.next = null;
		fast.next = head;
		return node;
	}

	public String addBinary(String a, String b) {
		if (a.isEmpty() || b.isEmpty())
			return a.isEmpty() ? b : a;
		int carry = 0;
		int i = a.length() - 1;
		int j = b.length() - 1;

		String res = "";
		while (i >= 0 && j >= 0) {
			int sum = a.charAt(i) - '0' + b.charAt(j) - '0' + carry;
			carry = sum / 2;
			sum = sum % 2;
			res = sum + res;
			i--;
			j--;
		}

		while (i >= 0) {
			int sum = a.charAt(i) - '0' + carry;
			carry = sum / 2;
			sum = sum % 2;
			res = sum + res;
			i--;
		}

		while (j >= 0) {
			int sum = b.charAt(j) - '0' + carry;
			carry = sum / 2;
			sum = sum % 2;
			res = sum + res;
			j--;
		}
		if (carry == 1)
			return "1" + res;
		return res;
	}

	public String addBinary2(String a, String b) {
		int m = a.length();
		int n = b.length();
		String res = "";
		int carry = 0;

		int len = Math.max(m, n);

		for (int i = 0; i < len; i++) {
			int p = 0, q = 0;
			if (i < m)
				p = a.charAt(m - 1 - i) - '0';
			if (i < n)
				q = b.charAt(n - 1 - i) - '0';
			int sum = p + q + carry;
			carry = sum / 2;
			sum = sum % 2;
			res = sum + res;
		}
		if (carry == 1)
			return "1" + res;
		return res;
	}

	public String addBinary3(String a, String b) {
		int i = a.length() - 1;
		int j = b.length() - 1;
		StringBuilder sb = new StringBuilder();
		int carry = 0;
		while (i >= 0 || j >= 0) {
			int num1 = i >= 0 ? a.charAt(i--) - '0' : 0;
			int num2 = j >= 0 ? b.charAt(j--) - '0' : 0;
			int sum = num1 + num2 + carry;
			carry = sum / 2;
			sum = sum % 2;
			sb.insert(0, sum);
		}
		if (carry == 1)
			sb.insert(0, 1);
		return sb.toString();
	}

	// Reservoir sampling: randomly choosing a sample of k items from a list S
	// containing n items, where n is either a very large or unknown number.
	// Typically n is large enough that the list doesn't fit into main memory
	// Algorithm:
	// array R[k]; // result
	// integer i, j;
	//
	// // fill the reservoir array
	// for each i in 1 to k do
	// R[i] := S[i]
	// done;
	//
	// // replace elements with gradually decreasing probability
	// for each i in k+1 to length(S) do
	// j := random(1, i); // important: inclusive range
	// if j <= k then
	// R[j] := S[i]
	// fi
	// done
	public int[] randomSelectK(int[] A, int k) {
		int[] res = new int[k];
		if (A.length < k)
			return res;
		for (int i = 0; i < k; i++) {
			res[i] = A[i];
		}

		for (int i = k; i < A.length; i++) {
			int index = (int) (Math.random() * (i + 1));
			if (index < k)
				res[index] = A[i];
		}
		return res;
	}

	public RandomListNode copyRandomList2(RandomListNode head) {
		if (head == null)
			return null;
		HashMap<RandomListNode, RandomListNode> map = new HashMap<RandomListNode, RandomListNode>();

		RandomListNode node = head;
		while (node != null) {
			RandomListNode copy = new RandomListNode(node.label);
			map.put(node, copy);
			node = node.next;
		}

		node = head;
		while (node != null) {
			RandomListNode copy = map.get(node);
			copy.next = map.get(node.next);
			copy.random = map.get(node.random);
			node = node.next;
		}
		return map.get(head);
	}

	public RandomListNode copyRandomList3(RandomListNode head) {
		if (head == null)
			return null;

		RandomListNode cur = head;
		while (cur != null) {
			RandomListNode next = cur.next;
			RandomListNode copy = new RandomListNode(cur.label);
			cur.next = copy;
			copy.next = next;
			cur = next;
		}

		cur = head;
		while (cur != null) {
			if (cur.random != null)
				cur.next.random = cur.random.next;
			cur = cur.next;
		}
		RandomListNode clone = head.next;
		cur = head;
		RandomListNode cur1 = clone;
		while (cur != null) {
			cur.next = cur1.next;
			cur = cur.next;
			if (cur != null)
				cur1.next = cur.next;
			cur1 = cur1.next;
		}
		return clone;
	}

	// lintcode binary search may have duplicates
	public int binarySearch2(int[] nums, int target) {
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

	// insert node into BST
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

	// find kth largest element in a list
	public int kthLargestElement(int k, ArrayList<Integer> numbers) {
		// write your code here
		int n = numbers.size() - 1;
		return quickSelect(0, n, k, numbers);
	}

	public int quickSelect(int left, int right, int k,
			ArrayList<Integer> numbers) {
		int pivot = left;
		int i = left + 1;
		int j = right;

		while (i <= j) {
			while (i <= j && numbers.get(i) >= numbers.get(pivot))
				i++;
			while (i <= j && numbers.get(j) <= numbers.get(pivot))
				j--;
			if (i < j)
				swap(i, j, numbers);
		}
		swap(pivot, j, numbers);

		if (j == k - 1)
			return numbers.get(j);
		else if (j > k - 1)
			return quickSelect(left, j - 1, k, numbers);
		else
			return quickSelect(j + 1, right, k, numbers);
	}

	public void swap(int i, int j, ArrayList<Integer> numbers) {
		int t = numbers.get(i);
		numbers.set(i, numbers.get(j));
		numbers.set(j, t);
	}

	public static int leftLeavesSum(TreeNode root) {
		if (root == null)
			return 0;
		int res = 0;
		if (isLeaf(root.left))
			res += root.left.val;
		else
			res += leftLeavesSum(root.left);
		res += leftLeavesSum(root.right);
		return res;
	}

	public static boolean isLeaf(TreeNode root) {
		if (root == null)
			return false;
		if (root.left == null && root.right == null)
			return true;
		return false;
	}

	public double binarySQRT(int x) {
		double low = 0;
		double high = x;
		double guess = (low + high) / 2;
		System.out.println("guess is " + guess);
		while (Math.abs(guess * guess - x) > 1e-5) {
			if (guess * guess > x)
				high = guess;
			else
				low = guess;
			guess = (low + high) / 2;
		}
		return guess;
	}

	public ListNode deleteEveryOther(ListNode head) {
		if (head == null || head.next == null)
			return head;
		ListNode cur = head;
		while (cur != null && cur.next != null) {
			System.out.println("cur value is " + cur.val);
			ListNode pnext = cur.next.next;
			cur.next = pnext;
			cur = cur.next;
		}
		return head;
	}

	public ListNode commonNodes(ListNode l1, ListNode l2) {
		if (l1 == null || l2 == null)
			return null;
		if (l1.val == l2.val) {
			ListNode node = new ListNode(l1.val);
			node.next = commonNodes(l1.next, l2.next);
			return node;
		} else if (l1.val < l2.val)
			return commonNodes(l1.next, l2);
		else
			return commonNodes(l1, l2.next);
	}

	public ListNode addBefore(ListNode head, int target, int value) {
		ListNode tmp = head;

		while (tmp != null) {
			if (tmp.val == target) {
				ListNode n = new ListNode(tmp.val);
				n.next = tmp.next;
				tmp.val = value;
				tmp.next = n;
				return head;
			}
			tmp = tmp.next;
		}
		return null;
	}

	public void rotate2(int[] nums, int k) {
		int n = nums.length;
		k = k % n;
		if (k == 0)
			return;
		reverse(nums, 0, n - 1);
		reverse(nums, 0, k - 1);
		reverse(nums, k, n - 1);
	}

	public void reverse(int[] nums, int i, int j) {
		while (i < j) {
			int t = nums[i];
			nums[i] = nums[j];
			nums[j] = t;
			i++;
			j--;
		}
	}

	public void rotate(int[] nums, int k) {
		int n = nums.length;
		k = k % n;
		if (k == 0)
			return;
		for (int i = 0; i < k; i++)
			rightRotateOne(nums);
	}

	public void rightRotateOne(int[] nums) {
		int n = nums.length;
		int t = nums[n - 1];
		for (int i = n - 1; i > 0; i--) {
			nums[i] = nums[i - 1];
		}
		nums[0] = t;
	}

	// public List<String> replaceString(String s){
	// List<String> res=new ArrayList<String>();
	// replaceString(s.toCharArray(), 0, res);
	// return res;
	// }
	//
	// public void replaceString(char[] s, int cur, List<String> res){
	// if(cur==s.length){
	// res.add(new String(s));
	// return;
	// }
	// if(s[cur]!='?')
	// replaceString(s, cur+1, res);
	// else{
	// for(char c='0';c<='1';c++){
	// s[cur]=c;
	// replaceString(s, cur+1, res);
	// s[cur]='?';
	// }
	// }
	// }

	public List<String> replaceString(String s) {
		List<String> res = new ArrayList<String>();
		replaceString(s, 0, "", res);
		return res;
	}

	public void replaceString(String s, int cur, String sol, List<String> res) {
		if (cur == s.length()) {
			res.add(sol);
			return;
		}
		if (s.charAt(cur) != '?')
			replaceString(s, cur + 1, sol + s.charAt(cur), res);
		else {
			for (char c = '0'; c <= '1'; c++) {
				replaceString(s, cur + 1, sol + c, res);
			}
		}

	}

	// uber all possible palindromes
	public boolean is_palindrome(String s) {
		if (s.length() < 2)
			return true;
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (!map.containsKey(c))
				map.put(c, 1);
			else
				map.put(c, map.get(c) + 1);
		}

		int odd = 0;
		Iterator<Character> it = map.keySet().iterator();
		while (it.hasNext()) {
			char c = it.next();
			if (map.get(c) % 2 != 0)
				odd++;
		}
		return odd < 2;
	}

	public String halfString(String s) {
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (!map.containsKey(c))
				map.put(c, 1);
			else
				map.put(c, map.get(c) + 1);
		}
		StringBuilder sb = new StringBuilder();
		Iterator<Character> it = map.keySet().iterator();
		while (it.hasNext()) {
			char c = it.next();
			if (map.get(c) % 2 == 0) {
				for (int i = 0; i < map.get(c) / 2; i++)
					sb.append(c);
			}

		}
		return sb.toString();
	}

	public List<String> generate_AllPalindromes(String s) {
		List<String> res = new ArrayList<String>();
		boolean midC = false;
		char mid = ' ';
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (!map.containsKey(c))
				map.put(c, 1);
			else
				map.put(c, map.get(c) + 1);
		}
		int odd = 0;
		StringBuilder sb = new StringBuilder();
		Iterator<Character> it = map.keySet().iterator();
		while (it.hasNext()) {
			char c = it.next();
			if (map.get(c) % 2 == 0) {
				for (int i = 0; i < map.get(c) / 2; i++)
					sb.append(c);
			} else {
				midC = true;
				mid = c;
				odd++;
			}

		}
		if (odd > 1)
			return res;
		boolean[] visited = new boolean[sb.length()];
		generateAllPalindromes(sb.toString(), 0, "", visited, res, mid);
		return res;
	}

	// public String reverseString(String s){
	// String rev="";
	// for(int i=s.length()-1;i>=0;i--)
	// rev +=s.charAt(i);
	// return rev;
	// }

	public void generateAllPalindromes(String s, int cur, String sol,
			boolean[] visited, List<String> res, char mid) {
		if (cur == s.length()) {
			System.out.println("now is " + sol);
			String rev = reverseString(sol);
			if (mid != ' ')
				rev = rev + mid + sol;
			else
				rev = rev + sol;
			res.add(rev);
			// return;
		}

		System.out.println("s is " + s);
		for (int i = 0; i < s.length(); i++) {
			if (!visited[i]) {
				if (i != 0 && s.charAt(i) == s.charAt(i - 1) && !visited[i - 1])
					continue;
				visited[i] = true;
				generateAllPalindromes(s, cur + 1, sol + s.charAt(i), visited,
						res, mid);
				visited[i] = false;
			}
		}
	}

	public int reverseBits(int n) {
		int res = 0;
		for (int i = 0; i < 32; i++) {
			int b = n & 1;
			res = res * 2 + b;
			n >>= 1;
		}
		return res;
	}

	public int hammingWeight(int n) {
		int count = 0;
		for (int i = 0; i < 32; i++) {
			count += n & 1;
			n >>= 1;
		}
		return count;
	}

	public boolean isWellFormed(String s, String brackets) {
		HashMap<Character, Character> map = new HashMap<Character, Character>();
		for (int i = 0; i < brackets.length(); i += 2) {
			map.put(brackets.charAt(i), '(');
			map.put(brackets.charAt(i + 1), ')');
		}

		Stack<Character> stk = new Stack<Character>();

		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (c == '(' || c == '[' || map.containsKey(c) && map.get(c) == '(')
				stk.push(c);
			else if (c == ')' || c == ']' || map.containsKey(c)
					&& map.get(c) == ')') {
				if (stk.isEmpty())
					return false;
				else {
					if (c == ')' && stk.peek() == '(' || c == ']'
							&& stk.peek() == '[' || map.containsKey(c)
							&& map.get(stk.peek()) == '(')
						stk.pop();
					else
						return false;
				}
			}
		}
		return stk.isEmpty();
	}

	public static void reverseKeys(TreeNode root) {
		if (root == null)
			return;
		TreeNode left = root.left;
		root.left = root.right;
		root.right = left;

		reverseKeys(root.left);
		reverseKeys(root.right);
	}

	public List<Event> findLastEventOfSession(List<Event> events) {
		List<Event> res = new ArrayList<Event>();
		if (events.size() == 0)
			return res;
		HashMap<Integer, Event> map = new HashMap<Integer, Event>();
		for (int i = 0; i < events.size(); i++) {
			Event e = events.get(i);
			if (map.containsKey(e.sessionId)) {
				if (e.timestamp > map.get(e.sessionId).timestamp)
					map.put(e.sessionId, e);
			} else
				map.put(e.sessionId, e);
		}
		Iterator<Integer> it = map.keySet().iterator();
		while (it.hasNext()) {
			int id = it.next();
			res.add(map.get(id));
		}
		return res;
	}

	public int square(int n) {
		if (n == 0)
			return 0;
		if (n < 0)
			n = -n;
		int x = n >> 1;
		if ((n & 1) == 0)
			return 4 * square(x);
		else
			return 4 * square(x) + 4 * x + 1;
	}

	// magine you have a special keyboard with the following keys:
	// Key 1: Prints 'A' on screen
	// Key 2: (Ctrl-A): Select screen
	// Key 3: (Ctrl-C): Copy selection to buffer
	// Key 4: (Ctrl-V): Print buffer on screen appending it
	// after what has already been printed.
	//
	// If you can only press the keyboard for N times (with the above four
	// keys), write a program to produce maximum numbers of A's. That is to
	// say, the input parameter is N (No. of keys that you can press), the
	// output is M (No. of As that you can produce).

	public int findOptimal(int n) {
		if (n < 7)
			return n;
		int[] screen = new int[n];

		for (int i = 1; i < 7; i++)
			screen[i - 1] = i;

		for (int i = 7; i <= n; i++) {
			screen[i - 1] = 0;

			for (int b = i - 3; b > 0; b--) {
				int cur = (i - b - 1) * screen[b - 1];
				if (cur > screen[i - 1])
					screen[i - 1] = cur;
			}
		}
		return screen[n - 1];
	}

	// Find the minimum cost to reach destination using a train
	public int minCost(int cost[][]) {
		int n = cost.length;
		int[] dist = new int[n];
		for (int i = 1; i < n; i++)
			dist[i] = Integer.MAX_VALUE;

		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				if (dist[j] > dist[i] + cost[i][j])
					dist[j] = dist[i] + cost[i][j];
			}
		}
		return dist[n - 1];
	}

	// Count number of islands where every island is row-wise and column-wise
	// separated
	// We can check if a ‘X’ is top left or not by checking following
	// conditions.
	// 1) A ‘X’ is top of rectangle if the cell just above it is a ‘O’
	// 2) A ‘X’ is leftmost of rectangle if the cell just left of it is a ‘O’

	public static int countIslands(char[][] mat) {
		int m = mat.length;
		if (m == 0)
			return 0;
		int n = mat[0].length;
		int count = 0;

		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (mat[i][j] == 'X') {
					if ((i == 0 || mat[i - 1][j] == 'O')
							&& (j == 0 || mat[i][j - 1] == 'O'))
						count++;
				}
			}
		}
		return count;
	}

	// Find maximum depth of nested parenthesis in a string

	public int maxDepParenthesis(String s) {
		int n = s.length();
		if (n < 2)
			return 0;
		int count = 0;
		int max = 0;
		for (int i = 0; i < n; i++) {
			if (s.charAt(i) == '(') {
				count++;
				if (count > max)
					max = count;
			} else if (s.charAt(i) == ')') {
				if (count > 0)
					count--;
				else
					return -1;
			}
		}
		if (count != 0)
			return -1;
		return max;
	}

	// A simple method to rearrange 'arr[0..n-1]' so that 'arr[j]'
	// becomes 'i' if 'arr[i]' is 'j'
	public void rearrangeNaive(int arr[]) {
		int n = arr.length;

		// Create an auxiliary array of same size
		int[] temp = new int[n];

		// Store result in temp[]
		for (int i = 0; i < n; i++)
			temp[arr[i]] = i;

		// Copy temp back to arr[]
		for (int i = 0; i < n; i++)
			arr[i] = temp[i];
	}

	// function takes an infinite size array and a key to be
	// searched and returns its position if found else -1.
	// We don't know size of arr[] and we can assume size to be
	// infinite in this function.
	public int findPos(int arr[], int key) {
		int l = 0, h = 1;
		int val = arr[0];

		// Find h to do binary search
		while (val < key) {
			l = h; // store previous high
			h = 2 * h; // double high index
			val = arr[h]; // update new val
		}

		// at this point we have updated low and high indices,
		// thus use binary search between them
		return binarySearch(arr, l, h, key);
	}

	public int binarySearch(int[] A, int low, int high, int key) {
		while (high >= low) {
			int mid = (low + high) / 2;
			if (A[mid] == key)
				return mid;
			else if (A[mid] > key)
				high = mid - 1;
			else
				low = mid + 1;
		}
		return -1;
	}

	// Count number of ways to reach a given score in a game
	// Consider a game where a player can score 3 or 5 or 10 points in a move.
	// Given a total score n, find number of ways to reach the given score.

	public int countWays(int n) {
		int[] table = new int[n + 1];
		table[0] = 1;

		for (int i = 3; i <= n; i++)
			table[i] += table[i - 3];
		for (int i = 5; i <= n; i++)
			table[i] += table[i - 5];
		for (int i = 10; i <= n; i++)
			table[i] += table[i - 10];

		return table[n];
	}

	// Find length of period in decimal value of 1/n

	// Given a positive integer n, find the period in decimal value of 1/n.
	// Period in decimal value is number of digits (somewhere after decimal
	// point) that keep repeating.

	public int getPeriod(int n) {
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		int rem = 1;
		int cur = 1;

		while (true) {
			rem = (rem * 10) % n;
			if (map.containsKey(rem))
				return cur - map.get(rem);
			map.put(rem, cur);
			cur++;
		}
		// return Integer.MAX_VALUE;
	}

	public int getPeriod2(int n) {
		int rem = 1;
		int count = 0;
		// Find the (n+1)th remainder after decimal point
		// in value of 1/n
		for (int i = 0; i <= n; i++)
			rem = (rem * 10) % n;

		int d = rem;

		// Count the number of remainders before next
		// occurrence of (n+1)'th remainder 'd'

		do {
			rem = (rem * 10) % n;
			count++;
		} while (rem != d);
		return count;
	}

	public int vertexCover(TreeNode root) {
		if (root == null)
			return 0;
		if (root.left == null && root.right == null)
			return 0;
		int size_incl = 1 + vertexCover(root.left) + vertexCover(root.right);
		int size_excl = 0;
		if (root.left != null)
			size_excl += 1 + vertexCover(root.left.left)
					+ vertexCover(root.left.right);
		if (root.right != null)
			size_excl += 1 + vertexCover(root.right.left)
					+ vertexCover(root.right.right);
		return Math.min(size_incl, size_excl);
	}

	// Longest Even Length Substring such that Sum of First and Second Half is
	// same

	public int findLength(String s) {
		int n = s.length();
		int maxLen = 0;

		for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j += 2) {// choose even length
				int length = j - i + 1;
				int leftsum = 0;
				int rightsum = 0;
				for (int k = 0; k < length / 2; k++) {
					leftsum += s.charAt(i + k) - '0';
					rightsum += s.charAt(i + k + length / 2) - '0';
				}
				if (leftsum == rightsum && length > maxLen)
					maxLen = length;
			}
		}
		return maxLen;
	}

	public int findLength2(String s) {
		int n = s.length();
		int[][] sum = new int[n][n];

		for (int i = 0; i < n; i++)
			sum[i][i] = s.charAt(i) - '0';
		int maxLen = 0;
		for (int len = 2; len <= n; len++) {
			for (int i = 0; i < n - len + 1; i++) {
				int j = i + len - 1;
				int k = len / 2;

				sum[i][j] = sum[i][j - k] + sum[j - k + 1][j];

				if (len % 2 == 0 && sum[i][j - k] == sum[j - k + 1][j]
						&& len > maxLen)
					maxLen = len;
			}
		}
		return maxLen;
	}

	public static TreeNode getClosestNode(TreeNode root, int target) {
		if (root == null)
			return null;
		TreeNode cur = root;
		TreeNode res = null;
		int minDis = Integer.MAX_VALUE;
		while (cur != null) {
			int dist = Math.abs(cur.val - target);
			if (dist < minDis) {
				minDis = dist;
				res = cur;
			}
			if (dist == 0)
				break;
			if (cur.val > target)
				cur = cur.left;
			else
				cur = cur.right;
		}
		return res;
	}

	// Find the longest substring with k unique characters in a given string

	public String kUniques(String s, int k) {
		if (s.length() < k)
			return "";
		int[] count = new int[26];
		int u = 0;
		int start = 0;

		int maxLen = 1;
		int window_start = 0;
		for (int i = 0; i < s.length(); i++) {
			if (count[s.charAt(i) - 'a'] == 0)
				u++;
			count[s.charAt(i) - 'a']++;
			if (u == k) {
				int len = i - start + 1;
				if (len > maxLen) {
					maxLen = len;
					window_start = start;
				}
			}
			while (u > k) {
				count[s.charAt(start) - 'a']--;
				if (count[s.charAt(start) - 'a'] == 0)
					u--;
				start++;
			}
		}
		if (u < k)
			return "";
		String res = s.substring(window_start, window_start + maxLen);
		System.out.println("Max sustring is is " + res + " with length "
				+ maxLen);
		return s.substring(window_start, window_start + maxLen);
	}

	public static TreeNode kthLargest(TreeNode root, int k) {
		if (root == null)
			return null;
		int[] count = { 0 };
		return kthLargestUtil(root, k, count);
	}

	public static TreeNode kthLargestUtil(TreeNode root, int k, int[] count) {
		if (root == null || count[0] > k)
			return null;
		kthLargestUtil(root.right, k, count);
		count[0]++;
		if (count[0] == k) {
			System.out.println(k + "th largest is " + root.val);
			return root;
		}
		kthLargestUtil(root.left, k, count);
		return null;
	}

	// 其主要用了两个基本表达式：
	// x^y //执行加法，不考虑进位。
	// (x&y)<<1 //进位操作
	// 令x=x^y ；y=(x&y)<<1
	// 进行迭代，每迭代一次进位操作右面就多一位0，最多需要“加数二进制位长度”次迭代就没有进位了，此时x^y的值就是结果。

	public int aplusb(int a, int b) {
		// Click submit, you will get Accepted!
		while (b != 0) {
			int carry = a & b;
			a = a ^ b;
			b = carry << 1;
		}
		return a;
	}

	// Group multiple occurrence of array elements ordered by first occurrence
	// Input: arr[] = {5, 3, 5, 1, 3, 3}
	// Output: {5, 5, 3, 3, 3, 1}

	public int[] orderedGroup(int arr[]) {
		if (arr.length < 2)
			return arr;
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < arr.length; i++) {
			if (!map.containsKey(arr[i]))
				map.put(arr[i], 1);
			else
				map.put(arr[i], map.get(arr[i]) + 1);
		}
		int[] res = new int[arr.length];
		int j = 0;
		for (int i = 0; i < arr.length; i++) {
			if (map.containsKey(arr[i])) {
				int count = map.get(arr[i]);
				for (int k = 0; k < count; k++)
					res[j++] = arr[i];
				map.remove(arr[i]);
			}
		}
		return res;
	}

	public int maxProfit(int k, int[] prices) {
		int n = prices.length;
		if (n < 2)
			return 0;
		int[][] global = new int[n][k + 1];
		int[][] local = new int[n][k + 1];

		for (int i = 1; i < n; i++) {
			int diff = prices[i] - prices[i - 1];
			for (int j = 1; j < k; j++) {
				local[i][j] = Math.max(local[i - 1][j] + diff,
						global[i - 1][j - 1] + Math.max(0, diff));
				global[i][j] = Math.max(local[i][j], global[i - 1][j]);
			}
		}
		return global[n - 1][k];
	}

	// Build smallest Number by Removing n digits from a given number
	public String buildSmallestNumber(String str, int n) {
		String[] res = { "" };
		buildSmallestNumberRec(str, n, res);
		return res[0];
	}

	public void buildSmallestNumberRec(String s, int n, String[] res) {
		if (n == 0) {
			res[0] += s;
			return;
		}
		if (n > s.length())
			return;
		int minIndex = 0;
		// The idea is based on the fact that a character among first (n+1)
		// characters must be there in resultant number.
		for (int i = 1; i <= n; i++) {
			if (s.charAt(i) < s.charAt(minIndex))
				minIndex = i;
		}

		res[0] += s.charAt(minIndex);
		buildSmallestNumberRec(s.substring(minIndex + 1), n - minIndex, res);

	}

	public boolean isCircular(String path) {
		int N = 0;
		int E = 1;
		int S = 2;
		int W = 3;

		int x = 0, y = 0;
		int dir = N;

		for (int i = 0; i < path.length(); i++) {
			char move = path.charAt(i);
			if (move == 'R')
				dir = (dir + 1) % 4;
			else if (move == 'L')
				dir = (4 + dir - 1) % 4;
			else {// move=='G'
				if (dir == N)
					y++;
				else if (dir == E)
					x++;
				else if (dir == S)
					y--;
				else
					x--;
			}
		}
		return x == 0 && y == 0;
	}

	public boolean twoSumRotated(int[] A, int target) {
		if (A.length < 2)
			return false;
		int index = -1;
		for (int i = 0; i < A.length - 1; i++) {
			if (A[i] > A[i + 1]) {
				index = i + 1;
			}
		}
		int i = index;
		int j = index - 1;

		while (i != j) {
			int sum = A[i] + A[j];
			if (sum == target) {
				return true;
			} else if (sum > target) {
				j = (A.length + j - 1) % A.length;
			} else
				i = (i + 1) % A.length;
		}
		return false;
	}

	public void wiggleSort(int[] A) {
		if (A == null || A.length < 2)
			return;
		int index = 0;
		while (index < A.length - 1) {
			swap(A, index, index + 1);
			index += 2;
		}
	}

	public void wiggleSort2(int[] arr) {
		// Input checking.
		if (arr == null || arr.length == 0)
			return;

		// Arrays.sort(arr).
		int index = 0;
		while (index < arr.length - 1) {
			swap(arr, index, index + 1);
			index += 2;
		}
	}

	TreeNode res = null;

	public TreeNode deepestNode(TreeNode root) {
		int[] maxlevel = { 0 };
		deepestLeafNode(root, 0, maxlevel);
		return res;
	}

	public void deepestLeafNode(TreeNode root, int level, int[] maxlevel) {
		if (root == null)
			return;
		if (root.left == null && root.right == null && level > maxlevel[0]) {
			res = root;
			maxlevel[0] = level;
			return;
		}
		deepestLeafNode(root.left, level + 1, maxlevel);
		deepestLeafNode(root.right, level + 1, maxlevel);
	}

	public TreeNode deepestLeftLeafNode(TreeNode root) {
		int[] maxlevel = { 0 };
		deepestLeftLeafNode(root, 0, maxlevel, false);
		return res;
	}

	public void deepestLeftLeafNode(TreeNode root, int level, int[] maxlevel,
			boolean isLeft) {
		if (root == null)
			return;
		if (isLeft && root.left == null && root.right == null
				&& level > maxlevel[0]) {
			res = root;
			maxlevel[0] = level;
			return;
		}
		deepestLeftLeafNode(root.left, level + 1, maxlevel, true);
		deepestLeftLeafNode(root.right, level + 1, maxlevel, false);
	}

	public List<String> findMissingRanges(int[] A, int lower, int upper) {
		List<String> res = new ArrayList<String>();
		if (A.length == 0) {
			if (lower == upper)
				res.add(lower + "");
			else {
				res.add(lower + "->" + upper);
			}
		} else {
			if (lower < A[0]) {
				if (A[0] - lower < 2)
					res.add(lower + "");
				else
					res.add(lower + "->" + (A[0] - 1));
			}

			for (int i = 1; i < A.length; i++) {
				if (A[i] - A[i - 1] == 2)
					res.add(A[i - 1] + 1 + "");
				else if (A[i] - A[i - 1] > 2) {
					res.add(A[i - 1] + 1 + "->" + (A[i] - 1));
				}
			}
			if (upper > A[A.length - 1]) {
				if (upper - A[A.length - 1] == 1)
					res.add(upper + "");
				else {
					res.add(A[A.length - 1] + 1 + "->" + upper);
				}
			}
		}
		return res;
	}

	public List<Integer> rightSideView(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		if (root == null)
			return res;
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		int curlevel = 0;
		int nextlevel = 0;
		que.offer(root);
		curlevel++;

		while (!que.isEmpty()) {
			TreeNode top = que.poll();
			curlevel--;
			if (top.left != null) {
				que.offer(top.left);
				nextlevel++;
			}
			if (top.right != null) {
				que.offer(top.right);
				nextlevel++;
			}
			if (curlevel == 0) {
				res.add(top.val);
				curlevel = nextlevel;
				nextlevel = 0;
			}
		}
		return res;

	}

	public List<Integer> rightSideView2(TreeNode root) {
		List<Integer> res = new ArrayList<Integer>();
		int[] maxlevel = { 0 };
		rightSideView(root, 1, maxlevel, res);
		return res;
	}

	public void rightSideView(TreeNode root, int level, int[] maxlevel,
			List<Integer> res) {
		if (root == null)
			return;
		if (level > maxlevel[0]) {
			res.add(root.val);
			maxlevel[0] = level;
		}
		rightSideView(root.right, level + 1, maxlevel, res);
		rightSideView(root.left, level + 1, maxlevel, res);
	}

	public int lengthOfLongestSubstringTwoDistinct(String s) {
		if (s.length() < 3)
			return s.length();
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		int start = 0;
		int maxLen = 0;

		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (map.containsKey(c))
				map.put(c, map.get(c) + 1);
			else if (map.size() < 2)
				map.put(c, 1);
			else {
				while (map.size() == 2 && start < i) {
					char t = s.charAt(start);
					int count = map.get(t);
					count--;
					if (count > 0)
						map.put(t, count);
					else
						map.remove(t);
					start++;
				}
				map.put(c, 1);
			}
			maxLen = Math.max(maxLen, i - start + 1);
		}
		return maxLen;
	}

	public int lengthOfLongestSubstringTwoDistinct2(String s) {
		if (s.length() < 3)
			return s.length();
		int maxLen = 0;
		int start = 0;
		int j = -1;
		for (int i = 1; i < s.length(); i++) {
			if (s.charAt(i) == s.charAt(i - 1))
				continue;
			if (j >= 0 && s.charAt(i) != s.charAt(j)) {
				maxLen = Math.max(maxLen, i - start);
				start = j + 1;
			}
			j = i - 1;
		}
		return Math.max(maxLen, s.length() - start);
	}

	public int lengthOfLongestSubstringKDistinct(String s, int k) {
		int start = 0;
		int max = 0;
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		for (int i = 0; i < s.length(); i++) {
			char c = s.charAt(i);
			if (map.containsKey(c))
				map.put(c, map.get(c) + 1);
			else if (map.size() < k)
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
			max = Math.max(max, i - start + 1);
		}
		return max;
	}

	public int rescheduleTask(List<Integer> tasks, int N) {
		int n = tasks.size();
		if (n < 2)
			return n;
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		int count = 0;
		for (int i = 0; i < n; i++) {
			int task = tasks.get(i);
			if (!map.containsKey(task)) {
				count++;
				map.put(task, count);
			} else {
				if (count - map.get(task) < N) {
					count += N - (count - map.get(task)) + 1;
				} else
					count++;
				map.put(task, count);
			}
		}
		return count;
	}

	// zenefits
	// String s1 =
	// "waeginsapnaabangpisebbasepgnccccapisdnfngaabndlrjngeuiogbbegbuoecccc";
	// String s2 = "a+b+c-";
	//
	// s2的形式是一个字母加上一个符号，正号代表有两个前面的字符，负号代表有四个，也就是说s2其实是"aabbcccc"，不考虑invalid。
	// 在s1中，找出连续或者不连续的s2，也就是说从s1中找出"aa....bb.....cccc"，abc顺序不能变，但是之间可以有零个或多个字符，返回共有多少个。在上面这个例子中，有四个。

	public int distinctSubsequences(String s1, String s2) {
		int n1 = s1.length();
		int n2 = s2.length();
		int[][] dp = new int[n1 + 1][n2 + 1];
		for (int i = 0; i <= n1; i++)
			dp[i][0] = 1;

		for (int i = 2; i <= n1; i++) {
			for (int j = 2; j <= n2; j++) {
				if (s2.charAt(j - 1) == '+') {
					String s = "";
					for (int k = 0; k < 2; k++) {
						s += s2.charAt(j - 2);
					}
					if (s1.substring(i - 2, i).equals(s))
						dp[i][j] = dp[i - 2][j] + dp[i - 2][j - 2];
					else
						dp[i][j] = dp[i - 1][j];
				} else if (s2.charAt(j - 1) == '-' && i >= 4) {
					String s = "";
					for (int k = 0; k < 4; k++) {
						s += s2.charAt(j - 2);
					}
					if (s1.substring(i - 4, i).equals(s))
						dp[i][j] = dp[i - 4][j] + dp[i - 4][j - 2];
					else
						dp[i][j] = dp[i - 1][j];
				}
			}
		}
		// for(int i=0;i<=n1;i++)
		// System.out.println(Arrays.toString(dp[i]));
		return dp[n1][n2];
	}

	public int distinctSubsequences2(String s1, String s2) {
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		StringBuilder sb1 = new StringBuilder();
		StringBuilder sb2 = new StringBuilder();
		for (int i = 0; i < s2.length() - 1; i += 2) {
			char c = s2.charAt(i);
			sb2.append(c);
			map.put(c, s2.charAt(i + 1) == '+' ? 1 : 3);
		}
		s2 = sb2.toString();

		for (int i = 0; i < s1.length(); i++) {
			char c = s1.charAt(i);
			if (map.containsKey(c)) {
				int count = map.get(c);
				if (isValid(s1, i, c, count))
					sb1.append(c);
			}
		}
		s1 = sb1.toString();

		int[][] dp = new int[s1.length() + 1][s2.length() + 1];
		for (int i = 0; i <= s1.length(); i++)
			dp[i][0] = 1;
		for (int i = 1; i <= s1.length(); i++) {
			for (int j = 1; j <= s2.length(); j++) {
				dp[i][j] = dp[i - 1][j];
				if (s1.charAt(i - 1) == s2.charAt(j - 1))
					dp[i][j] += dp[i - 1][j - 1];
			}
		}
		return dp[s1.length()][s2.length()];
	}

	public boolean isValid(String s, int i, char c, int count) {
		if (i + count >= s.length())
			return false;
		for (int j = 1; j <= count; j++) {
			if (s.charAt(i + j) != c)
				return false;
		}
		return true;
	}

	public int distinctSubsequencesRecur(String s1, String s2) {
		if (s2.length() == 0)
			return 1;
		if (s1.length() < s2.length())
			return 0;
		if (s2.charAt(1) == '+') {
			String s = "" + s2.charAt(0) + s2.charAt(0);
			if (s1.length() >= 2 && s1.substring(0, 2).equals(s))
				return distinctSubsequencesRecur(s1.substring(2), s2)
						+ distinctSubsequencesRecur(s1.substring(2),
								s2.substring(2));
			else
				return distinctSubsequencesRecur(s1.substring(1), s2);
		} else if (s2.charAt(1) == '-') {
			String s = "";
			for (int i = 0; i < 4; i++)
				s += s2.charAt(0);
			if (s1.length() >= 4 && s1.substring(0, 4).equals(s))
				return distinctSubsequencesRecur(s1.substring(4), s2)
						+ distinctSubsequencesRecur(s1.substring(4),
								s2.substring(2));
			else
				return distinctSubsequencesRecur(s1.substring(1), s2);
		} else
			return 0;
	}

	public int[][] findNearestPolice(int[][] board) {
		int n = board.length;
		if (n == 0)
			return null;
		int[][] res = new int[n][n];
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				res[i][j] = Integer.MAX_VALUE;
		}

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 1) {
					boolean[][] visited = new boolean[n][n];
					int count = 0;
					bfsFindPolice(board, i, j, visited, count, res);
				}
			}
		}

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 1)
					res[i][j] = 0;
				if (board[i][j] == 2 || res[i][j] == Integer.MAX_VALUE)
					res[i][j] = -1;
			}

		}
		return res;
	}

	public void bfsFindPolice(int[][] board, int i, int j, boolean[][] visited,
			int count, int[][] res) {
		int n = board.length;
		Queue<Pair> que = new LinkedList<Pair>();
		int curlevel = 0;
		int nextlevel = 0;

		que.add(new Pair(i, j));
		curlevel++;
		while (!que.isEmpty()) {
			Pair p = que.remove();
			// System.out.println(p);
			curlevel--;
			int row = p.first;
			int col = p.second;
			visited[row][col] = true;
			if (board[row][col] == 0 && res[row][col] > count)
				res[row][col] = count;
			if (row > 0 && !visited[row - 1][col] && board[row - 1][col] != 2) {
				que.add(new Pair(row - 1, col));
				nextlevel++;
			}
			if (row < n - 1 && !visited[row + 1][col]
					&& board[row + 1][col] != 2) {
				que.add(new Pair(row + 1, col));
				nextlevel++;
			}

			if (col > 0 && !visited[row][col - 1] && board[row][col - 1] != 2) {
				que.add(new Pair(row, col - 1));
				nextlevel++;
			}
			if (col < n - 1 && !visited[row][col + 1]
					&& board[row][col + 1] != 2) {
				que.add(new Pair(row, col + 1));
				nextlevel++;
			}

			if (curlevel == 0) {
				curlevel = nextlevel;
				nextlevel = 0;
				count++;
			}

		}
	}

	public void findNearestPolice2(int[][] board) {
		int n = board.length;

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] == 2) {
					int count = 0;
					boolean[][] visited = new boolean[n][n];
					bfsBoard(board, i, j, count, visited);
				}
			}
		}

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				if (board[i][j] < 0)
					board[i][j] = -board[i][j];
				else if (board[i][j] == 2)
					board[i][j] = 0;
				else if (board[i][j] == 0)
					board[i][j] = -1;
			}
			System.out.println(Arrays.toString(board[i]));
		}

	}

	public void bfsBoard(int[][] board, int i, int j, int count,
			boolean[][] visited) {
		int n = board.length;
		int t = n * i + j;
		Queue<Pair> que = new LinkedList<Pair>();
		int curlevel = 0;
		int nextlevel = 0;
		que.add(new Pair(i, j));
		curlevel++;
		while (!que.isEmpty()) {
			Pair p = que.remove();
			curlevel--;
			int r = p.first;
			int c = p.second;
			visited[r][c] = true;
			System.out.println(r + "," + c);
			if (board[r][c] == 1 || board[r][c] < 0)
				board[r][c] = board[r][c] == 1 ? -count : -1
						* Math.min(count, Math.abs(board[r][c]));
			if (r > 0 && board[r - 1][c] != 0 && !visited[r - 1][c]) {
				que.add(new Pair(r - 1, c));
				nextlevel++;
			}
			if (r < n - 1 && board[r + 1][c] != 0 && !visited[r + 1][c]) {
				que.add(new Pair(r + 1, c));
				nextlevel++;
			}
			if (c > 0 && board[r][c - 1] != 0 && !visited[r][c - 1]) {
				que.add(new Pair(r, c - 1));
				nextlevel++;
			}
			if (c < n - 1 && board[r][c + 1] != 0 && !visited[r][c + 1]) {
				que.add(new Pair(r, c + 1));
				nextlevel++;
			}

			if (curlevel == 0) {
				curlevel = nextlevel;
				nextlevel = 0;
				count++;
			}

		}
	}

	// zenefits
	public int flipBits(int[] bits) {
		for (int i = 0; i < bits.length; i++) {
			if (bits[i] == 1)
				bits[i] = -1;
			if (bits[i] == 0)
				bits[i] = 1;
		}
		System.out.println(Arrays.toString(bits));
		int sum = 0;
		int max = 0;
		// int beg=0;
		// int begIndex=0;
		// int endIndex=0;
		for (int i = 0; i < bits.length; i++) {
			sum += bits[i];
			if (sum > max) {
				// begIndex=beg;
				// endIndex=i;
				max = sum;
			}
			if (sum < 0) {
				sum = 0;
				// beg=i+1;
			}
		}
		for (int i = 0; i < bits.length; i++) {
			if (bits[i] == -1)
				max++;
		}
		return max;
	}

	public int bitFlip(int[] arr) {
		int maxCount = 0;
		int count = 0;
		for (int i = 0; i < arr.length; i++) {
			count = count < 0 ? 0 : count;
			if (arr[i] == 0)
				count++;
			else
				count--;
			maxCount = Math.max(maxCount, count);
		}
		for (int i = 0; i < arr.length; i++) {
			if (arr[i] == 1)
				maxCount++;
		}
		return maxCount;
	}

	public int totalPairOfDiffK(int[] nums, int k) {
		if (nums.length < 2)
			return 0;
		int count = 0;

		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < nums.length; i++)
			map.put(nums[i], i);

		for (int i = 0; i < nums.length; i++) {
			int num = nums[i];
			if (map.containsKey(num + k))
				count++;
			if (map.containsKey(num - k))
				count++;
			map.remove(num);
		}
		return count;
	}

	public void findCavity(char[][] board) {
		int n = board.length;
		if (n < 3)
			return;
		for (int i = 1; i < n - 1; i++) {
			for (int j = 1; j < n - 1; j++) {
				if (board[i][j] > board[i - 1][j]
						&& board[i][j] > board[i + 1][j]
						&& board[i][j] > board[i][j - 1]
						&& board[i][j] > board[i][j + 1])
					board[i][j] = 'X';
			}
		}
		for (int i = 0; i < n; i++) {
			System.out.println(Arrays.toString(board[i]));
		}

	}

	public TreeNode constructTreePreorder(int[] pre) {
		if (pre.length == 0)
			return null;
		TreeNode root = new TreeNode(pre[0]);
		Stack<TreeNode> stk = new Stack<TreeNode>();
		stk.push(root);
		TreeNode temp = null;
		for (int i = 1; i < pre.length; i++) {
			while (!stk.isEmpty() && pre[i] > stk.peek().val)
				temp = stk.pop();
			if (temp != null) {
				temp.right = new TreeNode(pre[i]);
				stk.push(temp.right);
			} else {
				stk.peek().left = new TreeNode(pre[i]);
				stk.push(stk.peek().left);
			}
		}
		return root;
	}

	public void findMissingRanges(int[] nums, int limit) {
		boolean[] seen = new boolean[limit];
		for (int i = 0; i < nums.length; i++) {
			if (nums[i] < limit)
				seen[nums[i]] = true;
		}

		int i = 0;
		while (i < limit) {
			if (!seen[i]) {
				int j = i + 1;
				while (j < limit && !seen[j])
					j++;
				if (i + 1 == j)
					System.out.println(i);
				else
					System.out.println(i + "->" + (j - 1));
				i = j;
			} else
				i++;
		}
	}

	public List<List<Integer>> printPathBy5(TreeNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> path = new ArrayList<Integer>();
		printPathUtil(root, path, res);
		return res;
	}

	public void printPathUtil(TreeNode root, List<Integer> path,
			List<List<Integer>> res) {
		if (root == null)
			return;
		path.add(root.val);
		if (root.val % 5 == 0) {
			res.add(new ArrayList<Integer>(path));
		}
		printPathUtil(root.left, path, res);
		printPathUtil(root.right, path, res);
		path.remove(path.size() - 1);
	}

	public String pushPopSequence(int[] A, int[] B) {
		Stack<Integer> stk = new Stack<Integer>();
		String s = "";
		int j = 0;
		for (int i = 0; i < A.length; i++) {
			int num = A[i];
			stk.push(num);
			s += "push" + num + "|";
			while (!stk.isEmpty() && stk.peek() == B[j]) {
				int t = stk.pop();
				s += "pop" + t + "|";
				j++;
			}
		}
		return s;
	}

	public void findRepettion(int[] arr) {
		if (arr.length < 2)
			return;
		for (int i = 0; i < arr.length; i++) {
			int num = Math.abs(arr[i]);
			if (arr[num] >= 0)
				arr[num] = -arr[num];
			else
				System.out.print(num + " ");
		}

		for (int i = 0; i < arr.length; i++) {
			if (arr[i] < 0)
				arr[i] = -arr[i];
		}
		System.out.println(Arrays.toString(arr));
	}

	public boolean isValidPreorder(int[] preorder) {
		if (preorder.length < 2)
			return true;
		Stack<Integer> stk = new Stack<Integer>();
		int lowerBound = Integer.MAX_VALUE;
		for (int i = 0; i < preorder.length; i++) {
			if (lowerBound != Integer.MAX_VALUE && preorder[i] < lowerBound)
				return false;
			while (!stk.isEmpty() && stk.peek() < preorder[i])
				lowerBound = stk.pop();
			stk.push(preorder[i]);
		}
		return true;
	}

	public List<Integer> findIslandSize(int[][] ocean) {
		List<Integer> res = new ArrayList<Integer>();
		for (int i = 0; i < ocean.length; i++) {
			for (int j = 0; j < ocean[0].length; j++) {
				if (ocean[i][j] == 1) {
					int[] size = { 0 };
					dfsOcean(ocean, i, j, size);
					res.add(size[0]);
				}
			}
		}
		return res;
	}

	public void dfsOcean(int[][] ocean, int i, int j, int[] size) {
		if (i < 0 || i >= ocean.length || j < 0 || j >= ocean[0].length
				|| ocean[i][j] == 0 || ocean[i][j] == 2)
			return;
		if (ocean[i][j] == 1) {
			ocean[i][j] = 2;
			size[0]++;
		}
		dfsOcean(ocean, i + 1, j, size);
		dfsOcean(ocean, i - 1, j, size);
		dfsOcean(ocean, i, j - 1, size);
		dfsOcean(ocean, i, j + 1, size);
	}

	public TreeNode kthLargestNodeBST(TreeNode root, int k) {
		int size = getTreeSize(root);

		if (root == null || k <= 0 || k > size)
			return null;
		int rightSize = getTreeSize(root.right);
		if (k == rightSize + 1)
			return root;
		else if (k < rightSize + 1)
			return kthLargestNodeBST(root.right, k);
		return kthLargestNodeBST(root.left, k - rightSize - 1);
	}

	public int getTreeSize(TreeNode root) {
		if (root == null)
			return 0;
		return getTreeSize(root.left) + 1 + getTreeSize(root.right);
	}

	TreeNode head = null;
	TreeNode tail = null;

	public boolean twoSumBST(TreeNode root, int target) {
		if (root == null)
			return false;
		convertBST2DLL(root);

		while (head != tail) {
			int sum = head.val + tail.val;
			if (sum == target)
				return true;
			else if (sum > target)
				tail = tail.left;
			else
				head = head.right;
		}
		return false;
	}

	public void convertBST2DLL(TreeNode root) {
		if (root == null)
			return;
		convertBST2DLL(root.left);
		root.left = tail;
		if (tail != null)
			tail.right = root;
		else
			head = root;
		tail = root;
		convertBST2DLL(root.right);

	}

	// Find if there is a triplet in a Balanced BST that adds to zero
	public boolean isTripletPresent(TreeNode root) {
		if (root == null)
			return false;
		convertBST2DLL(root.left);

		while (head.right != tail && head.val < 0) {
			TreeNode left = head.right;
			TreeNode right = tail;
			while (left != right) {
				int sum = left.val + right.val + head.val;
				if (sum == 0)
					return true;
				if (sum > 0)
					right = right.left;
				else
					left = left.right;
			}
			head = head.right;
		}
		return false;
	}

	// last bit of (odd number & even number) is 0.
	// when m != n, There is at least a odd number and a even number, so the
	// last bit position result is 0.
	// Move m and n rigth a position.
	// Keep doing step 1,2,3 until m equal to n, use a factor to record the
	// iteration time.

	public int rangeBitwiseAnd(int m, int n) {
		if (m == 0)
			return 0;
		int moveFactor = 1;
		while (m != n) {
			m >>= 1;
			n >>= 1;
			moveFactor <<= 1;
		}
		return m * moveFactor;
	}

	public List<String> findRepeatedDnaSequences(String s) {
		List<String> res = new ArrayList<String>();
		HashSet<Integer> set = new HashSet<Integer>();
		HashSet<Integer> added = new HashSet<Integer>();

		for (int i = 10; i <= s.length(); i++) {
			String dna = s.substring(i - 10, i);
			int n = convert(dna);

			if (!added.contains(n)) {
				if (!set.contains(n)) {
					set.add(n);
				} else {
					added.add(n);
					res.add(dna);
				}
			}
		}
		return res;
	}

	public int convert(String s) {
		HashMap<Character, Integer> map = new HashMap<Character, Integer>();
		map.put('A', 0);
		map.put('C', 1);
		map.put('G', 2);
		map.put('T', 3);
		int res = 0;
		for (int i = 0; i < s.length(); i++) {
			res = res * 4 + map.get(s.charAt(i));
		}
		return res;
	}

	// zenefits
	public long buySellStocks(int[] prices) {
		int n = prices.length;
		if (n < 2)
			return 0;
		long max = 0;
		int maxPrice = prices[n - 1];
		for (int i = n - 2; i >= 0; i--) {
			if (prices[i] > maxPrice)
				maxPrice = prices[i];
			else
				max += maxPrice - prices[i];
		}
		return max;
	}

	// public int transform2GoodNodes(List<Integer> input){
	// HashMap<Integer, Integer> map=new HashMap<Integer, Integer>();
	// for(int i=0;i<input.size();i++){
	// int node=input.get(i);
	// map.put(i+1, node);
	// }
	//
	// HashMap<Integer, List<Integer>> clusters=new HashMap<Integer,
	// List<Integer>>();
	// Iterator<Integer> it=map.keySet().iterator();
	// while(it.hasNext()){
	// int node=it.next();
	// int cluster=map.get(node);
	// if(clusters.containsKey(cluster)){
	// clusters.get(cluster).add(node);
	// }
	// else{
	// List<Integer> list=new ArrayList<Integer>();
	// list.add(node);
	// clusters.put(cluster, list);
	// }
	// }
	//
	// HashMap<Integer, List<Integer>> res=new HashMap<Integer,
	// List<Integer>>();
	// it=clusters.keySet().iterator();
	// while(it.hasNext()){
	//
	// }
	// if(clusters.containsKey(1))
	// return clusters.size()-1;
	// return clusters.size();
	// }

	public int paintWays(int n) {
		int[] dp = new int[n + 1];
		dp[0] = 0;
		dp[1] = 2;
		dp[2] = 4;
		for (int i = 3; i <= n; i++) {
			dp[i] = dp[i - 1] + dp[i - 2];
		}
		return dp[n];
	}

	public int paintWays2(int n) {
		if (n <= 0)
			return 0;
		if (n == 1)
			return 2;
		if (n == 2)
			return 4;
		int lastTwoSame = 2;
		int lastTwoDif = 2;

		for (int i = 3; i <= n; i++) {
			int t = lastTwoSame;
			lastTwoSame = lastTwoDif;
			lastTwoDif = t + lastTwoDif;
		}
		return lastTwoDif + lastTwoSame;
	}

	public int findPeak(int[] A) {
		return findPeakUtil(A, 0, A.length - 1);
	}

	public int findPeakUtil(int[] A, int start, int end) {
		if (start > end)
			return -1;
		int mid = start + (end - start) / 2;
		if ((mid == 0 || A[mid] > A[mid - 1])
				&& (mid == end || A[mid] > A[mid + 1]))
			return mid;
		else if (mid > 0 && A[mid - 1] > A[mid])
			return findPeakUtil(A, start, mid - 1);
		return findPeakUtil(A, mid + 1, end);
	}

	public int longest01Strings(String[] strs, int m, int n) {
		int len = strs.length;
		int[][][] dp = new int[m + 1][n + 1][len + 1];

		for (int i = 0; i <= m; i++) {
			for (int j = 0; j <= n; j++) {

				for (int k = 1; k <= len; k++) {
					int count0 = 0;
					int count1 = 0;
					String s = strs[k - 1];
					for (int c = 0; c < s.length(); c++) {
						if (s.charAt(c) == '0')
							count0++;
						else
							count1++;
					}
					if (count0 <= i && count1 <= j)
						dp[i][j][k] = Math.max(dp[i][j][k - 1],
								dp[i - count0][j - count1][k - 1] + 1);
					else
						dp[i][j][k] = dp[i][j][k - 1];
				}
			}
		}
		return dp[m][n][len];
	}

	// 一个找最大值的题。给一个数组[1,1,2,1]，然后用+ * （）三个操作求出这个数组的最大值，这个题返回6
	public int maxValueOfArray(int[] num) {
		int n = num.length;
		int[][] dp = new int[n][n];
		for (int i = 0; i < n; i++) {
			dp[i][i] = num[i];
			for (int j = i - 1; j >= 0; j--) {
				int diff = i - j;
				int tmax = 0;
				for (int k = 0; k < diff; k++) {
					tmax = Math.max(
							tmax,
							Math.max(dp[i][i - k] + dp[i - k - 1][j], dp[i][i
									- k]
									* dp[i - k - 1][j]));
				}
				dp[i][j] = tmax;
			}
		}

		for (int i = 0; i < n; i++) {
			System.out.println(Arrays.toString(dp[i]));
		}
		return dp[n - 1][0];
	}

	public int maxValueOfArray2(int[] num) {
		int n = num.length;
		int[][] dp = new int[n][n];
		for (int i = 0; i < n; i++)
			dp[i][i] = num[i];
		for (int len = 1; len <= n; len++) {
			for (int i = 0; i < n - len + 1; i++) {
				int j = i + len - 1;
				for (int k = i; k < j; k++) {
					dp[i][j] = Math.max(
							dp[i][j],
							Math.max(dp[i][k] + dp[k + 1][j], dp[i][k]
									* dp[k + 1][j]));
				}
			}
		}

		for (int i = 0; i < n; i++) {
			System.out.println(Arrays.toString(dp[i]));
		}
		return dp[0][n - 1];
	}

	public ListNode removeElements(ListNode head, int val) {
		if (head == null)
			return null;
		ListNode dummy = new ListNode(0);
		dummy.next = head;
		ListNode pre = dummy;
		ListNode cur = head;
		while (cur != null) {
			if (cur.val == val)
				pre.next = cur.next;
			else
				pre = pre.next;
			cur = cur.next;
		}
		return dummy.next;
	}

	public boolean isHappy(int n) {
		HashSet<Integer> set = new HashSet<Integer>();
		while (n != 1) {
			set.add(n);
			int sum = 0;
			while (n != 0) {
				int t = n % 10;
				sum += t * t;
				n /= 10;
			}
			n = sum;
			if (set.contains(n))
				return false;
			System.out.println(n);
		}
		return true;
	}

	public List<List<String>> findFriends(List<Pair> friendPairs) {
		List<List<String>> res = new ArrayList<List<String>>();
		HashMap<String, List<String>> map = new HashMap<String, List<String>>();
		Queue<String> que = new LinkedList<String>();
		Set<String> visited = new HashSet<String>();
		for (int i = 0; i < friendPairs.size(); i++) {
			Pair p = friendPairs.get(i);
			if (!map.containsKey(p.f1)) {
				List<String> friends = new ArrayList<String>();
				friends.add(p.f2);
				map.put(p.f1, friends);
			} else {
				map.get(p.f1).add(p.f2);
			}

			if (!map.containsKey(p.f2)) {
				List<String> friends = new ArrayList<String>();
				friends.add(p.f1);
				map.put(p.f2, friends);
			} else {
				map.get(p.f2).add(p.f1);
			}
		}

		for (int i = 0; i < friendPairs.size(); i++) {
			Pair p = friendPairs.get(i);
			List<String> network = new ArrayList<String>();
			if (!visited.contains(p.f1)) {
				network.add(p.f1);
				visited.add(p.f1);
				que.add(p.f1);
				while (!que.isEmpty()) {
					String name = que.remove();
					if (!visited.contains(name)) {
						network.add(name);
						visited.add(name);
					}
					List<String> fs = map.get(name);
					for (int j = 0; j < fs.size(); j++) {
						if (!visited.contains(fs.get(j))) {
							que.add(fs.get(j));
						}
					}
				}
				res.add(network);
			}
		}
		return res;
	}

	class Triple {
		int arrayIdx;
		int index;
		int val;

		public Triple(int arrayIdx, int index, int val) {
			this.arrayIdx = arrayIdx;
			this.index = index;
			this.val = val;
		}
	}

	public int[] mergeNSortedArrays(int[][] sortedArrays) {
		int n = sortedArrays.length;
		int total = 0;
		PriorityQueue<Triple> que = new PriorityQueue<Triple>(n,
				new Comparator<Triple>() {

					@Override
					public int compare(Triple o1, Triple o2) {
						// TODO Auto-generated method stub
						return o1.val - o2.val;
					}

				});
		for (int i = 0; i < n; i++) {
			total += sortedArrays[i].length;
			que.add(new Triple(i, 0, sortedArrays[i][0]));
		}
		int[] res = new int[total];
		int k = 0;
		while (!que.isEmpty()) {
			Triple t = que.poll();
			res[k++] = t.val;
			if (t.index + 1 < sortedArrays[t.arrayIdx].length)
				que.add(new Triple(t.arrayIdx, t.index + 1,
						sortedArrays[t.arrayIdx][t.index + 1]));

		}
		System.out.println(res.length);
		System.out.println(Arrays.toString(res));
		return res;
	}

	// 输入是一个 N*N的矩阵，代表地势高度。如果下雨水流只能流去比他矮或者一样高的地势。
	// 矩阵左边和上边是太平洋，右边和下边是大西洋。求出所有的能同时流到两个大洋的点。

	// Pacific: ~
	// Atlantic: *
	// ~ ~ ~ ~ ~ ~ ~
	// ~ 1 2 2 3 (5) *
	// ~ 3 2 3 (4) (4) *
	// ~ 2 4 (5) 3 1 *
	// ~ (6) (7) 1 4 5 *
	// ~ (5) 1 1 2 4 *
	// * * * * * * *

	public List<Point> flowing_water(int[][] mat) {
		int m = mat.length;
		int n = mat[0].length;
		boolean[][] visited_pac = new boolean[m][n];

		for (int i = 0; i < m; i++) {
			visited_pac[i][0] = true;
			search(i, 0, visited_pac, mat);
		}

		for (int i = 0; i < n; i++) {
			visited_pac[0][i] = true;
			search(0, i, visited_pac, mat);
		}

		boolean[][] visited_atl = new boolean[m][n];

		for (int i = 0; i < m; i++) {
			visited_atl[i][n - 1] = true;
			search(i, n - 1, visited_atl, mat);
		}

		for (int i = 0; i < n; i++) {
			visited_atl[m - 1][i] = true;
			search(m - 1, i, visited_atl, mat);
		}

		List<Point> res = new ArrayList<Point>();
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (visited_pac[i][j] && visited_atl[i][j])
					res.add(new Point(i, j));
			}
		}
		return res;
	}

	public void search(int i, int j, boolean[][] visited, int[][] mat) {
		int[][] dirs = { { 0, 1 }, { 0, -1 }, { 1, 0 }, { -1, 0 } };

		for (int k = 0; k < 4; k++) {
			int new_x = i + dirs[k][0];
			int new_y = j + dirs[k][1];
			if (new_x < 0 || new_x >= mat.length || new_y < 0
					|| new_y >= mat[0].length || visited[new_x][new_y])
				continue;
			if (mat[new_x][new_y] < mat[i][j])
				continue;
			visited[new_x][new_y] = true;
			search(new_x, new_y, visited, mat);
		}
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
				if (minHeap.isEmpty() || minHeap.size() == maxHeap.size()
						|| maxHeap.size() > minHeap.size())
					res.add(maxHeap.peek());
				else
					res.add(minHeap.peek());
				// if (maxHeap.isEmpty())
				// res.add(minHeap.peek());
				// else if (minHeap.isEmpty())
				// res.add(maxHeap.peek());
				// else if (minHeap.size() == maxHeap.size()) {
				// res.add(maxHeap.peek());
				// } else if (maxHeap.size() > minHeap.size())
				// res.add(maxHeap.peek());
				// else
				// res.add(minHeap.peek());

				if (minHeap.contains(nums[i - k]))
					minHeap.remove(nums[i - k]);
				else
					maxHeap.remove(nums[i - k]);
			}

			if (minHeap.size() == maxHeap.size()) {
				if (!minHeap.isEmpty() && nums[i] > minHeap.peek()) {
					maxHeap.offer(minHeap.poll());
					minHeap.offer(nums[i]);
				} else
					maxHeap.offer(nums[i]);

			} else {
				if (maxHeap.size() > minHeap.size()) {
					if (nums[i] < maxHeap.peek()) {
						minHeap.offer(maxHeap.poll());
						maxHeap.offer(nums[i]);
					} else
						minHeap.offer(nums[i]);
				} else {
					if (nums[i] > minHeap.peek()) {
						maxHeap.offer(minHeap.poll());
						minHeap.offer(nums[i]);
					} else
						maxHeap.offer(nums[i]);
				}
			}
		}

		if (maxHeap.isEmpty())
			res.add(minHeap.peek());
		else if (minHeap.isEmpty())
			res.add(maxHeap.peek());
		else if (minHeap.size() == maxHeap.size())
			res.add(maxHeap.peek());
		else if (maxHeap.size() > minHeap.size())
			res.add(maxHeap.peek());
		else
			res.add(minHeap.peek());
		return res;
	}

	// public ArrayList<Integer> medianSlidingWindow2(int[] nums, int k) {
	// // write your code here
	// ArrayList<Integer> res=new ArrayList<Integer>();
	// int index=0;
	// if(k%2==0)
	// index=k/2-1;
	// else
	// index=k/2;
	// for(int i=0;i+k-1<nums.length;i++){
	// int median=findMedian(nums, i, i+k-1, index);
	// res.add(median);
	// }
	// return res;
	// }
	//
	// public int findMedian(int[] nums, int beg, int end, int k){
	// int pivot=beg;
	// int i=beg+1;
	// int j=end;
	// while(i<=j){
	// while(i<=j&&nums[i]<=nums[pivot])
	// i++;
	// while(i<=j&&nums[j]>nums[pivot])
	// j--;
	// if(i<j){
	// swap(nums, i, j);
	// i++;
	// j--;
	// }
	// }
	// swap(nums, i, j);
	// if(j==k)
	// return nums[j];
	// else if(j>k)
	// return findMedian(nums, beg, j-1, k);
	// else
	// return findMedian(nums, j+1, end, k);
	//
	// }

	// Input: I want to get a cup of water
	// Output: I wnat to get a cup of wtear.
	// （每个单词首尾字符不变，中间的字符打乱，也就是说每次输入会得到不同的输出）

	public String shuffleWords(String input) {
		String[] strs = input.split(" ");
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < strs.length; i++) {
			String s = strs[i];
			if (s.length() > 3) {
				s = shuffle(s);
			}
			sb.append(s + " ");
		}
		return sb.toString().trim();
	}

	public String shuffle(String s) {
		char[] ch = s.toCharArray();
		int n = s.length();
		for (int i = 1; i < n - 1; i++) {
			int index = (int) (Math.random() * (n - 1 - i)) + i;
			char c = ch[i];
			ch[i] = ch[index];
			ch[index] = c;
		}
		return new String(ch);
	}

	// zenefits
	// String input (M)
	// String pattern (N)

	// # output me the number of substrings in input that is an anagram of
	// pattern

	// input：abcba
	// pattern：abc
	// ~> 2
	public int numOfAnagramSubstrings(String M, String N) {
		int m = M.length();
		int n = N.length();
		if (m < n)
			return 0;
		int[] needToFind = new int[256];
		for (int i = 0; i < N.length(); i++) {
			needToFind[N.charAt(i)]++;
		}
		int count = 0;
		int[] hasFound = new int[256];
		int windowStart = 0;
		for (int i = 0; i < m; i++) {
			char c = M.charAt(i);
			if (needToFind[c] == 0) {
				hasFound = new int[256];
				windowStart = i + 1;
				continue;
			}
			hasFound[c]++;
			if (hasFound[c] > needToFind[c]) {
				while (hasFound[M.charAt(windowStart)] > needToFind[M
						.charAt(windowStart)]) {
					hasFound[M.charAt(windowStart)]--;
					windowStart++;
				}
			}
			if (i - windowStart + 1 == n) {
				count++;
				hasFound[M.charAt(windowStart)]--;
				windowStart++;
			}
		}
		return count;
	}

	public int countStr(String s, String p) {
		if (s == null || p == null || p.length() > s.length()) {
			return 0;
		}
		HashMap<Character, Integer> dict = new HashMap<Character, Integer>();
		for (int i = 0; i < p.length(); i++) {
			if (dict.containsKey(p.charAt(i))) {
				dict.put(p.charAt(i), dict.get(p.charAt(i)) + 1);
			} else {
				dict.put(p.charAt(i), 1);
			}
		}
		int num = 0;
		int left = 0;
		int count = 0;
		for (int right = 0; right < s.length(); right++) {
			if (dict.containsKey(s.charAt(right))) {
				dict.put(s.charAt(right), dict.get(s.charAt(right)) - 1);
				if (dict.get(s.charAt(right)) >= 0) {
					count++;
				}
				while (count == p.length()) {
					if ((right - left + 1) == p.length()) {
						num++;
					}
					if (dict.containsKey(s.charAt(left))) {
						dict.put(s.charAt(left), dict.get(s.charAt(left)) + 1);
						if (dict.get(s.charAt(left)) > 0) {
							count--;
						}
					}
					left++;
				}
			}
		}
		return num;
	}

	public boolean isIsomorphic2(String s, String t) {
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

	public int countPrimes(int n) {
		if (n < 3)
			return 0;

		boolean[] A = new boolean[n];
		for (int i = 0; i < n; i++) {
			A[i] = true;
		}

		for (int i = 2; i < Math.sqrt(n) + 1; i++) {
			if (A[i]) {
				for (int j = i * i; j < n; j += i) {
					A[j] = false;
				}
			}
		}
		int count = 0;
		for (int i = 2; i < n; i++) {
			if (A[i])
				count++;
		}
		return count;
	}

	// There are ‘n’ ticket windows in the railway station, ith window has ai
	// tickets available. Price of a ticket is equal to the number of tickets
	// remaining in that window at that time. When ‘m’ tickets have been sold,
	// what’s the maximum amount of money the railway station can earn?
	// e.g.
	// INPUT: n=2, m=4
	// a1=2 , a2=5
	// OUTPUT: 14(2nd window sold 4 tickets so 5+4+3+2).

	public int sellTicket(int[] tickets, int m) {
		if (m == 0)
			return 0;
		int count = 0;
		int total = 0;
		int n = tickets.length - 1;
		Arrays.sort(tickets);
		for (int i = n; i >= 0; i--) {
			while (tickets[i] >= tickets[i - 1] && count < m) {
				total += tickets[i];
				tickets[i]--;
				count++;
			}
			if (count == m)
				return total;
		}
		return total;
	}

	public int evaluateExpression(String[] expression) {
		Deque<String> ops = new ArrayDeque<String>();
		Queue<Integer> vals = new LinkedList<Integer>();

		for (int i = 0; i < expression.length; i++) {
			String token = expression[i];
			if (Character.isDigit(token.charAt(0))) {
				vals.offer(Integer.parseInt(token));
			} else if (token.equals("[")) {
				Queue<Integer> que = new LinkedList<Integer>();
				String op = expression[++i];
				ops.offer(op);
				i++;
				while (i < expression.length
						&& (!expression[i].equals("]") && !expression[i]
								.equals("["))) {
					que.add(Integer.parseInt(expression[i]));
					i++;
				}
				int t = que.poll();
				while (!que.isEmpty()) {
					t = applyOp(t, que.poll(), op);
				}

				vals.offer(t);
				if (expression[i].equals("]"))
					ops.pollLast();
				i--;
			} else if (token.equals("+") || token.equals("-")
					|| token.equals("*") || token.equals("/")) {
				ops.offer(token);
			}
		}

		if (vals.isEmpty() && !ops.isEmpty() && ops.peek().equals("*"))
			return 1;
		else if (vals.isEmpty())
			return 0;
		int res = vals.poll();

		while (!vals.isEmpty())
			res = applyOp(res, vals.poll(), ops.peek());
		// return vals.pop();
		return res;
	}

	public boolean isOp(String op) {
		if (op.equals("+") || op.equals("-") || op.equals("*")
				|| op.equals("/"))
			return true;
		return false;
	}

	public int applyOp(int val1, int val2, String op) {
		if (op.equals("+"))
			return val1 + val2;
		else if (op.equals("-"))
			return val1 - val2;
		else if (op.equals("*"))
			return val1 * val2;
		return val1 / val2;
	}

	// Returns true if there is a triplet with following property
	// A[i]*A[i] = A[j]*A[j] + A[k]*[k]
	// Note that this function modifies given array
	public boolean isTriplet(int arr[]) {
		if (arr.length < 3)
			return false;
		for (int i = 0; i < arr.length; i++) {
			arr[i] = arr[i] * arr[i];
		}
		Arrays.sort(arr);

		for (int i = arr.length - 1; i >= 0; i--) {
			int beg = 0;
			int end = i - 1;
			while (beg < end) {
				int sum = arr[beg] + arr[end];
				if (sum == arr[i])
					return true;
				else if (sum < arr[i])
					beg++;
				else
					end--;
			}
		}
		return false;
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

	public int findKthLargest(int[] nums, int k) {
		return quickSelect(0, nums.length - 1, nums, k);
	}

	public int quickSelect(int left, int right, int[] nums, int k) {
		int pivot = left;
		int beg = left + 1;
		int end = right;

		while (beg <= end) {
			while (beg <= end && nums[beg] > nums[pivot])
				beg++;
			while (beg <= end && nums[end] <= nums[pivot])
				end--;
			if (beg < end) {
				swap(beg, end, nums);
				beg++;
				end--;
			}
		}
		swap(pivot, end, nums);
		if (end == k - 1)
			return nums[end];
		else if (end > k - 1)
			return quickSelect(left, end - 1, nums, k);
		else
			return quickSelect(end + 1, right, nums, k);
	}

	public void swap(int i, int j, int[] nums) {
		int t = nums[i];
		nums[i] = nums[j];
		nums[j] = t;
	}

	public int rob(int[] nums) {
		if (nums.length == 1)
			return nums[0];
		return Math.max(rob(0, nums.length - 2, nums),
				rob(1, nums.length - 1, nums));
	}

	public int rob(int lo, int hi, int[] nums) {
		int inclusive = 0;
		int exclusive = 0;

		for (int j = lo; j <= hi; j++) {
			int i = inclusive;
			int e = exclusive;
			inclusive = e + nums[j];
			exclusive = Math.max(e, i);
		}
		return Math.max(inclusive, exclusive);
	}

	public int countNodes1(TreeNode root) {
		if (root == null)
			return 0;
		int left = getLeftHeight(root.left);
		int right = getRightHeight(root.right);
		System.out.println("height is " + left + ", " + right);
		if (left == right)
			return (int) (Math.pow(2, left + 1)) - 1;
		return 1 + countNodes1(root.left) + countNodes1(root.right);
	}

	public int getLeftHeight(TreeNode root) {
		if (root == null)
			return 0;
		TreeNode left = root;
		int h = 0;
		while (left != null) {
			h++;
			left = left.left;
		}
		return h;
	}

	public int getRightHeight(TreeNode root) {
		if (root == null)
			return 0;
		TreeNode right = root;
		int h = 0;
		while (right != null) {
			h++;
			right = right.right;
		}
		return h;
	}

	public boolean containsNearbyDuplicate(int[] nums, int k) {
		if (nums.length < 2)
			return false;
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		int dis = Integer.MAX_VALUE;
		for (int i = 0; i < nums.length; i++) {
			if (map.containsKey(nums[i])) {
				int preIdx = map.get(nums[i]);
				dis = Math.min(dis, i - preIdx);
				if (dis <= k)
					return true;
			}
			map.put(nums[i], i);

		}
		return false;
	}

	public boolean containsNearbyAlmostDuplicate(int[] nums, int k, int t) {
		if (nums.length < 2)
			return false;
		TreeMap<Long, Integer> map = new TreeMap<Long, Integer>();
		for (int i = 0; i < nums.length; i++) {
			if (i > k)
				map.remove((long) nums[i - k - 1]);
			long val = (long) nums[i];
			Long greater = map.ceilingKey(val);
			if (greater != null && greater - val <= t)
				return true;
			Long smaller = map.floorKey(val);
			if (smaller != null && val - smaller <= t)
				return true;
			map.put((long) nums[i], i);
		}
		return false;
	}

	public int computeArea(int A, int B, int C, int D, int E, int F, int G,
			int H) {
		int area1 = (C - A) * (D - B);
		int area2 = (G - E) * (H - F);
		int overlapWidth = 0;
		if (E <= C && G > A) {
			int leftBar = E < A ? A : E;
			int rightBar = G > C ? C : G;
			overlapWidth = rightBar - leftBar;
		}

		int overlapHeight = 0;
		if (F <= D && H > B) {
			int downBar = F < B ? B : F;
			int upBar = H > D ? D : H;
			overlapHeight = upBar - downBar;
		}
		return area1 + area2 - overlapWidth * overlapHeight;
	}

	public int computeArea2(int A, int B, int C, int D, int E, int F, int G,
			int H) {
		int area1 = (C - A) * (D - B);
		int area2 = (G - E) * (H - F);

		int overlapRegion = overlap(A, B, C, D, E, F, G, H);
		return area1 + area2 - overlapRegion;
	}

	private int overlap(int A, int B, int C, int D, int E, int F, int G, int H) {
		int h1 = Math.max(A, E);
		int h2 = Math.min(C, G);
		int h = h2 - h1;

		int v1 = Math.max(B, F);
		int v2 = Math.min(D, H);
		int v = v2 - v1;

		if (h <= 0 || v <= 0)
			return 0;
		else
			return h * v;
	}

	public List<List<Integer>> combinationSum3(int k, int n) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		combinationSumUtil(0, k, n, 0, sol, res, 1);
		return res;
	}

	public void combinationSumUtil(int dep, int maxDep, int n, int cursum,
			List<Integer> sol, List<List<Integer>> res, int cur) {
		if (dep > maxDep || cursum > n)
			return;
		if (dep == maxDep && cursum == n) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}

		for (int i = cur; i <= 9; i++) {
			cursum += i;
			sol.add(i);
			combinationSumUtil(dep + 1, maxDep, n, cursum, sol, res, i + 1);
			cursum -= i;
			sol.remove(sol.size() - 1);
		}
	}
// calculator I
	public int calculate(String s) {
		int res=0;
        Stack<Integer> stk=new Stack<Integer>();
        int sign=1;
        for(int i=0;i<s.length();i++){
            char c=s.charAt(i);
            if(c>='0'&&c<='9'){
                int cur=c-'0';
                while(i+1<s.length() && Character.isDigit(s.charAt(i+1))){
                    cur=cur*10+s.charAt(++i)-'0';
                }
                res+=sign*cur;
            }
            else if(c=='-')
                sign=-1;
            else if(c=='('){
                stk.push(res);
                res=0;
                stk.push(sign);
                sign=1;
            }
            else if(c==')'){
                res=res*stk.pop()+stk.pop();
                sign=1;
            }
            else 
                sign=1;

        }
        return res;
	}
	
	// CALCULATOR II
	public int calculate2(String s) {
        if(s.length()==0)
        	return 0;
        Stack<Integer> operands=new Stack<Integer>();
        Stack<Character> operators=new Stack<Character>();
        
        for(int i=0;i<s.length();i++){
        	char c=s.charAt(i);
        	if(Character.isDigit(c)){
            	int cur=c-'0';
            	while(i+1<s.length()&&Character.isDigit(s.charAt(i+1))){
            		cur=cur*10+s.charAt(i+1)-'0';
            		i++;
            	}
            	if(!operators.isEmpty()&&(operators.peek()=='*'||operators.peek()=='/')){
            		char op=operators.pop();
            		int operand=operands.pop();
            		if(op=='*')
            			operands.push(operand*cur);
            		else
            			operands.push(operand/cur);
            	}
            	else
            		operands.push(cur);
        	}
        	else if(c==' ')
        		continue;
        	else
        		operators.push(c);
        				
        }

        if(operands.isEmpty())
        	return 0;
        Collections.reverse(operands);
        Collections.reverse(operators);
        int res=operands.pop();
        
        while(!operators.isEmpty()&&!operands.isEmpty()){
        	int operand=operands.pop();
        	char operator=operators.pop();
        	if(operator=='+')
        		res+=operand;
        	else
        		res-=operand;
        }
        return res;
    }

	public List<String> summaryRanges(int[] nums) {
		List<String> res = new ArrayList<String>();
		if (nums.length == 0)
			return res;
		// String beg=""+nums[0];
		// int pre=nums[0];
		// for(int i=1;i<nums.length;i++){
		// if(nums[i]==pre+1)
		// pre=nums[i];
		// else{
		// int start=Integer.parseInt(beg);
		// if(pre==start)
		// res.add(beg);
		// else
		// res.add(beg+"->"+pre);
		// beg=""+nums[i];
		// pre=nums[i];
		// }
		// }
		// int start=Integer.parseInt(beg);
		// if(pre==start)
		// res.add(beg);
		// else
		// res.add(beg+"->"+pre);
		// return res;
		int s = 0, e = 0;

		while (e < nums.length) {
			if (e + 1 < nums.length && nums[e + 1] == nums[e] + 1)
				e++;
			else {
				if (nums[s] == nums[e])
					res.add("" + nums[s]);
				else {
					res.add(nums[s] + "->" + nums[e]);
				}
				s = ++e;
			}
		}
		return res;
	}

	public List<Integer> majorityElement(int[] nums) {
		List<Integer> res = new ArrayList<Integer>();
		int first = 0;
		int second = 0;
		int c1 = 0;
		int c2 = 0;

		for (int i = 0; i < nums.length; i++) {
			if (c1 == 0)
				first = nums[i];
			if (c2 == 0 && nums[i] != first)
				second = nums[i];
			if (nums[i] == first)
				c1++;
			else if (nums[i] == second)
				c2++;
			else {
				c1--;
				c2--;
			}
		}

		c1 = c2 = 0;
		for (int i = 0; i < nums.length; i++) {
			if (nums[i] == first)
				c1++;
			if (nums[i] == second)
				c2++;
		}
		if (c1 > nums.length / 3)
			res.add(first);
		if (second != first && c2 > nums.length / 3)
			res.add(second);

		return res;
	}

	// public int equilibrium(int[] A){
	// if(A.length<3)
	// return -1;
	// int left=0;
	// int right=A.length-1;
	// int sum1=0, sum2=0;
	//
	// while(left<right){
	// if(sum1<sum2)
	// sum1+=A[left++];
	// else if(sum1>sum2)
	// sum2+=A[right--];
	// else{
	// sum1+=A[left++];
	// sum2+=A[right--];
	// }
	// System.out.println(sum1+", " +sum2);
	// }
	// if(sum1==sum2)
	// return left;
	// else
	// return -1;
	// }

	public int equilibrium(int[] A) {
		int sum = 0;
		for (int i = 0; i < A.length; i++)
			sum += A[i];
		int leftSum = 0;
		for (int i = 0; i < A.length; i++) {
			sum -= A[i];
			if (leftSum == sum)
				return i;
			leftSum += A[i];
		}
		return -1;
	}

	public int[] productExceptSelf(int[] nums) {
		int n = nums.length;
		int[] left = new int[n];
		int[] right = new int[n];
		left[0] = 1;
		for (int i = 1; i < n; i++) {
			left[i] = left[i - 1] * nums[i - 1];
		}

		right[n - 1] = 1;
		for (int i = n - 2; i >= 0; i--) {
			right[i] = right[i + 1] * nums[i + 1];
		}

		int[] res = new int[n];
		for (int i = 0; i < n; i++) {
			res[i] = left[i] * right[i];
		}
		return res;
	}

	public TreeNode lowestCommonAncestorIterative(TreeNode root, TreeNode p,
			TreeNode q) {
		if (root == null || p == null || q == null)
			return null;
		if (p == root || q == root)
			return root;

		TreeNode cur = root;
		while (cur != null) {
			if (cur.val > q.val && cur.val > p.val)
				cur = cur.left;
			else if (cur.val < p.val && cur.val < q.val)
				cur = cur.right;
			else
				break;
		}
		return cur;
	}

	public TreeNode lowestCommonAncestorRecur(TreeNode root, TreeNode p,
			TreeNode q) {
		if (root == null)
			return null;
		if (root.val > p.val && root.val > q.val)
			return lowestCommonAncestorRecur(root.left, p, q);
		else if (root.val < p.val && root.val < q.val)
			return lowestCommonAncestorRecur(root.right, p, q);
		return root;
	}

	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
		if (root == null || p == null || q == null)
			return null;
		if (p == root || q == root)
			return root;
		TreeNode left = lowestCommonAncestor(root.left, p, q);
		TreeNode right = lowestCommonAncestor(root.right, p, q);
		if (left != null && right != null)
			return root;
		return left == null ? right : left;
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
						int v1 = res1.get(j);
						int v2 = res2.get(k);
						if (c == '+')
							res.add(v1 + v2);
						else if (c == '-')
							res.add(v1 - v2);
						else
							res.add(v1 * v2);
					}
				}
			}
		}
		if (res.size() == 0) {
			res.add(Integer.parseInt(input));
		}
		return res;
	}

	public List<Integer> diffWaysToComputeDP(String input) {
		Map<String, List<Integer>> map = new HashMap<String, List<Integer>>();
		return diffWaysComputeUtil(input, map);
	}

	public List<Integer> diffWaysComputeUtil(String input,
			Map<String, List<Integer>> map) {
		List<Integer> res = new ArrayList<Integer>();

		for (int i = 0; i < input.length(); i++) {
			char c = input.charAt(i);
			List<Integer> res1, res2;
			if (c == '+' || c == '-' || c == '*') {
				String sub1 = input.substring(0, i);
				String sub2 = input.substring(i + 1);

				if (map.containsKey(sub1))
					res1 = map.get(sub1);
				else
					res1 = diffWaysComputeUtil(sub1, map);

				if (map.containsKey(sub2))
					res2 = map.get(sub2);
				else
					res2 = diffWaysComputeUtil(sub2, map);

				for (int j = 0; j < res1.size(); j++) {
					for (int k = 0; k < res2.size(); k++) {
						int v1 = res1.get(j);
						int v2 = res2.get(k);
						if (c == '+')
							res.add(v1 + v2);
						else if (c == '-')
							res.add(v1 - v2);
						else
							res.add(v1 * v2);
					}
				}

			}
		}
		if (res.size() == 0) {
			res.add(Integer.parseInt(input));
		}
		map.put(input, res);
		return res;
	}

	// Count factorial numbers in a given range
	public int countFactorials(int low, int high) {
		int res = 0;
		int x = 1;
		int fact = 1;
		while (fact < low) {
			fact *= x;
			x++;
		}

		while (fact <= high) {
			res++;
			fact *= x;
			x++;
		}
		return res;
	}

	// Find sum of all elements in a matrix except the elements in row and/or
	// column of given cell?
	public void printSum(int[][] mat, Cell[] arr) {
		int sum = 0;
		int[] row = new int[mat.length];
		int[] col = new int[mat[0].length];

		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat[0].length; j++) {
				sum += mat[i][j];
				row[i] += mat[i][j];
				col[j] += mat[i][j];
			}
		}

		for (int i = 0; i < arr.length; i++) {
			int r = arr[i].r;
			int c = arr[i].c;
			System.out.println(sum - row[r] - col[c] + mat[r][c]);
		}
	}

	// Count distinct elements in every window of size k
	public void countDistinct(int arr[], int k) {
		Map<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < k; i++) {
			if (map.containsKey(arr[i]))
				map.put(arr[i], map.get(arr[i]) + 1);
			else
				map.put(arr[i], 1);
		}
		System.out.println(map.size());

		for (int i = k; i < arr.length; i++) {
			if (map.get(arr[i - k]) == 1)
				map.remove(arr[i - k]);
			else
				map.put(arr[i - k], map.get(arr[i - k]) - 1);
			if (map.containsKey(arr[i]))
				map.put(arr[i], map.get(arr[i]) + 1);
			else
				map.put(arr[i], 1);
			System.out.println(map.size());
		}
	}

	// Rotate Matrix Elements
	// Given a matrix, clockwise rotate elements in it.

	public void rotatematrix(int mat[][]) {
		int left = 0, right = mat[0].length - 1;
		int top = 0, bottom = mat.length - 1;

		while (true) {
			if (top + 1 > bottom || left + 1 > right)
				break;
			int pre = mat[top + 1][left];
			for (int i = left; i <= right; i++) {
				int cur = mat[top][i];
				mat[top][i] = pre;
				pre = cur;
			}
			top++;
			for (int i = top; i <= bottom; i++) {
				int cur = mat[i][right];
				mat[i][right] = pre;
				pre = cur;
			}
			right--;
			for (int i = right; i >= left; i--) {
				int cur = mat[bottom][i];
				mat[bottom][i] = pre;
				pre = cur;
			}
			bottom--;
			for (int i = bottom; i >= top; i--) {
				int cur = mat[i][left];
				mat[i][left] = pre;
				pre = cur;
			}
			left++;
		}

		for (int i = 0; i < mat.length; i++) {
			for (int j = 0; j < mat[0].length; j++) {
				System.out.print(mat[i][j]);
			}
			System.out.println();
		}
	}
	
	public List<String> binaryTreePaths(TreeNode root) {
        List<String> res=new ArrayList<String>();
        binaryTreePaths(root, "", res);
        return res;
    }
    
    public void binaryTreePaths(TreeNode root, String sol, List<String> res){
        if(root==null)
            return;
        if(sol.isEmpty())
            sol+=root.val;
        else
            sol+="->"+root.val;
        if(root.left==null&&root.right==null)
            res.add(sol);
        binaryTreePaths(root.left, sol, res);
        binaryTreePaths(root.right, sol, res);
    }

	public boolean evaluateExpression(String s, int target) {
		return evaluateExpression(s, target, 0);
	}

	public boolean evaluateExpression(String s, int target, int curVal) {
		if (s.length() == 0 && target == 0)
			return true;
		if (curVal > target)
			return false;
		if (curVal == target)
			return true;
		for (int i = 0; i <s.length(); i++) {
			int cur = convertToInt(s.substring(0, i+1));
			if (evaluateExpression(s.substring(i+1), target-curVal, cur))
				return true;
			if(curVal!=0&&evaluateExpression(s.substring(i+1), target/curVal, cur))
				return true;
		}
		return false;
	}

	public int convertToInt(String s) {
		if (s.length() == 0)
			return 0;
		int res = 0;
		for (int i = 0; i < s.length(); i++) {
			res = res * 10 + (s.charAt(i) - '0');
		}
		return res;
	}

	public String calculation(String str, int target) {
		String n = "no solution";
		if (str == null || str.length() == 0)
			return n;
		int len = str.length();
		int sum = 0;
		String result = "";
		List<String> list = new ArrayList<String>();
		rec(0, len, result, str, target, list);
		if (list.size() > 0)
			return list.get(0);
		else
			return n;
	}

	public void rec(int s, int len, String result, String str,
			int target, List<String> list) {
		if (s == len && calculateString(result) == target)
			list.add(new String(result));
		for (int i = s; i < len; i++) {
			String temp = str.substring(s, i + 1);
			if (i + 1 != len) {
				rec(i + 1, len, result + temp + "+", str, target, list);
				rec(i + 1, len, result + temp + "*", str, target, list);
			} else
				rec(i + 1, len, result + temp, str, target, list);
		}
	}

	public int calculateString(String s) {
		if (s == null)
			return 0;
		s = s.trim().replaceAll(" +", "");
		int length = s.length();
		int res = 0;
		long preVal = 0; // initial preVal is 0
		char sign = '+'; // initial sign is +
		int i = 0;
		while (i < length) {
			long curVal = 0;
			while (i < length && (int) s.charAt(i) <= 57
					&& (int) s.charAt(i) >= 48) { //
				curVal = curVal * 10 + (s.charAt(i) - '0');
				i++;
			}
			if (sign == '+') {
				res += preVal; // update res
				preVal = curVal;
			} else if (sign == '*') {
				preVal = preVal * curVal; // not update res, combine preVal & //
											// curVal and keep loop
			}
			if (i < length) { // getting new sign
				sign = s.charAt(i);
				i++;
			}
		}
		res += preVal;
		return res;

	}
	
//	reverse string的变种。 只reverse word不reverse punctuation。比如 "this,,,is.
//	a word" -> "word,,,a.is this"
	public String reverseString2(String s){
		if(s.length()<2)
			return s;
		System.out.println(s);
		char[] res=new char[s.length()];
		int i=0, j=s.length()-1;
		int start=0;
		while(i<s.length()&&!Character.isAlphabetic(s.charAt(i))){
			res[start++]=s.charAt(i++);
		}
		int beg=start;
		while(i<s.length()&&j>=0){		
			while(j>=0&&Character.isAlphabetic(s.charAt(j))){
				res[start++]=s.charAt(j--);
			}
			int end=start-1;
			while(beg<end){
				char c=res[beg];
				res[beg]=res[end];
				res[end]=c;
				beg++;
				end--;
			}
			while(j>=0&&!Character.isAlphabetic(s.charAt(j)))
				j--;
			while(i<s.length()&&Character.isAlphabetic(s.charAt(i)))
				i++;
			while(i<s.length()&&!Character.isAlphabetic(s.charAt(i))){
				res[start++]=s.charAt(i++);
			}
			beg=start;
			
		}
		
		return new String(res);
	}
	
	
	public int addDigits(int num) {
        while (num > 9) {
            num = getInt(num);
        }
        return num;
    }

    private int getInt(int num) {
        int result = 0;
        while (num >= 10) {
            result += num % 10;
            num /= 10;
        }
        result += num;
        return result;
    }
    
    public int addDigits2(int num) {
        int val=((num-1)/9)*9;
        return num-val;
    }
    
    public int[] maxSlidingWindow2(int[] nums, int k) {
    	if(k==0)
    		return new int[0];
        int[] res=new int[nums.length-k+1];
        ArrayDeque<Integer> q=new ArrayDeque<Integer>();
        
        for(int i=0;i<k;i++){
        	while(!q.isEmpty()&&nums[i]>=nums[q.peekLast()])
        		q.pollLast();
        	q.offerLast(i);
        }
        
        for(int i=k;i<nums.length;i++){
        	res[i-k]=nums[q.peekFirst()];
        	
        	while(!q.isEmpty()&&q.peekFirst()<=i-k)
        		q.pollFirst();
        	while(!q.isEmpty()&&nums[i]>=nums[q.peekLast()])
        		q.pollLast();
        	q.offerLast(i);
        }
        res[nums.length-k]=nums[q.peekFirst()];
        return res;
    }
    
    public int[] maxSlidingWindow3(int[] nums, int k) {
    	if(k==0)
    		return new int[0];
    	LinkedList<Integer> que=new LinkedList<Integer>();
    	int[] res=new int[nums.length-k+1];
    	
    	for(int i=0;i<nums.length;i++){
    		while(!que.isEmpty()&&nums[i]>=nums[que.getLast()])
    			que.pollLast();
    		que.add(i);
    		
    		if(i-que.getFirst()+1>k)
    			que.pollFirst();
    		if(i+1>=k)
    			res[i-k+1]=nums[que.getFirst()];
    	}
    	return res;
    }
    
    
    public List<String> findWords(char[][] board, String[] words) {
    	Set<String> res = new HashSet<String>();
        if(board==null || words==null || board.length==0 || words.length==0) return new ArrayList<String>(res);
        boolean[][] visited = new boolean[board.length][board[0].length]; 
        
        Trie trie = new Trie();
        for(String word : words) {
            trie.insert(word);
        }
        
        for(int i=0; i<board.length; i++) {
            for(int j=0; j<board[0].length; j++) {
            	findWordsUtil(board, trie, i, j, "",visited, res);
            }
        }
        return new ArrayList<String>(res);
    }
    
    public void findWordsUtil(char[][] board, Trie trie, int i, int j, String word, boolean[][] used, Set<String> res){
    	if(i<0 || i>=board.length || j<0 || j>=board[0].length || used[i][j])  return;
        
        word+=board[i][j];
        if(!trie.startsWith(word))
            return;
        if(trie.search(word))
           res.add(word);
	        used[i][j]=true;
	        findWordsUtil(board, trie, i+1, j, word, used, res);
	        findWordsUtil(board, trie, i-1, j, word, used, res);
	        findWordsUtil(board, trie, i, j+1, word, used, res);
	        findWordsUtil(board, trie, i, j-1, word, used, res);
	        used[i][j]=false;        
    }
    
//    public List<List<Integer>> fourSum(int[] nums, int target) {
//    	List<List<Integer>> res = new ArrayList<List<Integer>>();
//    	if(nums.length<4)
//    		return res;
//    	Map<Integer, List<Pair>> map=new HashMap<Integer, List<Pair>>();
//
//    	for(int i=0;i<nums.length-1;i++){
//    		for(int j=i+1;j<nums.length;j++){
//    			int sum=nums[i]+nums[j];
//    			Pair p=new Pair(nums[i], nums[j]);
//    			if(map.containsKey(sum)){			
//    				map.get(sum).add(p);
//    			}
//    			else{
//    				List<Pair> lst=new ArrayList<Pair>();
//    				lst.add(p);
//    				map.put(sum, lst);
//    			}
//    		}
//    	}
//    	
//    	Iterator<Integer> it=map.keySet().iterator();
//    	while(it.hasNext()){
//    		int key=it.next();
//    		int left=target-key;
//    		if(left!=key&&map.containsKey(left)){
//    			
//    		}
//    	}
//    }
    
	public List<List<Integer>> fourSum(int[] nums, int target) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (nums.length < 4)
			return res;
		Arrays.sort(nums);

		for (int i = 0; i < nums.length - 3; i++) {
			if (i == 0 || nums[i] != nums[i - 1]) {
				for (int j = i + 1; j < nums.length - 2; j++) {
					if (j == i + 1 || nums[j] != nums[j - 1]) {
						int beg = j + 1;
						int end = nums.length - 1;
						while (beg < end) {
							int sum = nums[i] + nums[j] + nums[beg] + nums[end];
							if (sum == target) {
								List<Integer> sol = new ArrayList<Integer>();
								sol.add(nums[i]);
								sol.add(nums[j]);
								sol.add(nums[beg]);
								sol.add(nums[end]);
								res.add(sol);
								beg++;
								end--;
								while (beg < end && nums[beg] == nums[beg - 1]) {
									beg++;
								}
								while (beg < end && nums[end] == nums[end + 1]) {
									end--;
								}
							} else if (sum > target) {
								while (beg < end && nums[end] == nums[end - 1]) {
									end--;
								}
								end--;
							} else {
								while (beg < end && nums[beg] == nums[beg + 1]) {
									beg++;
								}
								beg++;
							}
						}
					}
				}
			}
		}
		return res;
	}
    
    public int shortestDistance(String[] words, String word1, String word2) {
    	int min=words.length;
    	int index1=-1;
    	int index2=-1;
    	
    	for(int i=0;i<words.length;i++){
    		if(words[i].equals(word1))
    			index1=i;
    		if(words[i].equals(word2))
    			index2=i;
    		if(index1!=-1&&index2!=-1)
    			min=Math.min(min, Math.abs(index1-index2));
    	}
    	return min;
    }
    
    
//    Count number of paths with at-most k turns
//    Given a “m x n” matrix, count number of paths to reach bottom right from top left with maximum k turns allowed.
    public int countPaths(int[][] matrix, int i, int j, int k){
    	int[][][][] dp=new int[matrix.length][matrix[0].length][k+1][2];
    	for(int r=0;r<matrix.length;r++){
    		for(int c=0;c<matrix[0].length;c++){
    			for(int l=0;j<=k;l++)
    				for(int d=0;d<2;d++){
    					dp[r][c][l][d]=-1;
    				}
    		}
    	}
    	return countPathsUtil(dp, i, j, k, 0) + countPathsUtil(dp, i, j, k, 1);
    }
    
    public int countPathsUtil(int[][][][] dp, int i, int j, int k, int direc){
    	if(i<0||j<0||k<0)
    		return 0;
    	if(i==0&&j==0)
    		return 1;
    	if(k==0){
    		if(direc==0&&i==0)
    			return 1;
    		if(direc==1&&j==0)
    			return 1;
    	}
    	if(dp[i][j][k][direc]!=-1)
    		return dp[i][j][k][direc];
    	if(direc==0){
    		return dp[i][j][k][direc] = dp[i][j][k][0] + dp[i-1][j][k-1][1];
    	}
    	return dp[i][j][k][direc] = dp[i][j-1][k-1][0] + dp[i-1][j][k][1];
    		
    }
    
    
    public List<Float> filter(List<Float> lst, float v, float e){
    	List<Float> res=new ArrayList<Float>();
    	for(int i=0;i<res.size();i++){
    		Float val=lst.get(i);
    		if(Math.abs(val-v)<=e)
    			res.add(lst.get(i));
    	}
    	return res;
    }

	// longest common sequence 例如 A = {"A", "B", "C", "D"} B = {"D", "C", "A",
	// "B"}, result "AB", longest means the number of string not the sum of
	// length

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		Solution sol = new Solution();
		 Point p1 = new Point(2, 3);
		 Point p2 = new Point(3, 3);
		 Point p3 = new Point(-5, 3);
		 Point[] points = { p1, p2, p3 };
		
		 System.out.println(maxPoints(points));
		
		 System.out.println(longestValidParentheses("((()())(()()()"));
		 int[] A = { 3, 2, 5, 1, 4, 7, 9, 6, 8 };
		 findTriple(A);
		
		 int[] prices = { 3, 7, 4, 5, 8, 6, 9, 1, 7 };
		 profitStock(prices);
		 String s = "catsanddog";
		 Set<String> set = new HashSet<String>();
		 set.add("cat");
		 set.add("cats");
		 set.add("and");
		 set.add("sand");
		 set.add("dog");
		
		 System.out.println(wordBreak(s, set));
		
		 ListNode head = new ListNode(1);
		 ListNode node1 = new ListNode(2);
		 ListNode node2 = new ListNode(3);
		 ListNode node3 = new ListNode(4);
		 ListNode node4 = new ListNode(3);
		 ListNode node5 = new ListNode(2);
		 ListNode node6 = new ListNode(1);
		 head.next = node1;
		 node1.next = node2;
		 node2.next = node3;
		 // node3.next=node4;
		 // node4.next=node5;
		 // node5.next=node6;
		
		 ListNode reverseHead = reverseList2(head);
		 while (reverseHead != null) {
		 System.out.print(reverseHead.val + " ");
		 reverseHead = reverseHead.next;
		 }
		 System.out.println("***********");
		 ListNode reshead = mergeSortList(head);
		
		 while (reshead != null) {
		 System.out.print(reshead.val + " ");
		 reshead = reshead.next;
		 }
		 System.out.println();
		 System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~");
		
		 ListNode newhead = reverseKGroup(head, 3);
		
		 while (newhead != null) {
		 System.out.print(newhead.val + " ");
		 newhead = newhead.next;
		 }
		 System.out.println();
		 System.out.println(isPalindrome(head));
		 System.out.println(isPalindrome2(head));
		
		 skipMdeleteN(head, 1, 1);
		
		 while (head != null) {
		 System.out.print(head.val + " ");
		 head = head.next;
		 }
		 System.out.println();
		
		 TreeNode root = new TreeNode(5);
		 root.left = new TreeNode(2);
		 root.left.right = new TreeNode(4);
		 root.left.right.left = new TreeNode(3);
		 root.left.left = new TreeNode(1);
		
		
		 root.right = new TreeNode(8);
		 root.right.left = new TreeNode(6);
		 // root.right.left.left = new TreeNode(7);
		 root.right.right = new TreeNode(9);
		 root.right.right.right = new TreeNode(11);
		 // root.right.right.left.right = new TreeNode(13);
		 System.out.println("testing-testing-testing-----");
		 // System.out.println(sol.twoSumBST(root, 3));
		 // System.out.println(sol.isTripletPresent(root));
		 System.out.println(sol.kthLargestNodeBST(root, 1).val);
		 System.out.println(sol.kthLargestNodeBST(root, 2).val);
		 System.out.println(sol.kthLargestNodeBST(root, 3).val);
		 System.out.println(sol.kthLargestNodeBST(root, 4).val);
		 System.out.println(sol.kthLargestNodeBST(root, 5).val);
		 System.out.println(sol.kthLargestNodeBST(root, 6).val);
		 System.out.println(sol.kthLargestNodeBST(root, 7).val);
		 System.out.println(sol.kthLargestNodeBST(root, 8).val);
		 System.out.println(sol.kthLargestNodeBST(root, 9).val);
		 // System.out.println(sol.kthLargestNodeBST(root, 10).val);
		 System.out.println(sol.deepestNode(root).val);
		 // inorder(root);
		 // System.out.println();
		 kthLargest(root, 1);
		 kthLargest(root, 2);
		 kthLargest(root, 3);
		 kthLargest(root, 4);
		 System.out.println(kthLargest(root, 5).val);
		 kthLargest(root, 6);
		 kthLargest(root, 7);
		 kthLargest(root, 8);
		 kthLargest(root, 9);
		
		 // reverseKeys(root);
		 // inorder(root);
		 System.out.println(getClosestNode(root, 12).val);
		 System.out.println("xxxxxxxxxxxxxxxx");
		
		 System.out.println(leftLeavesSum(root));
		 System.out.println("xxxxxxxxxxxxxxxx");
		 System.out.println(visibleNodes(root));
		
		 System.out.println("~~~~~~~~~~~~~~~");
		 twoSumOfBST(root, 12);
		 System.out.println("~~~~~~~~~~~~~~~");
		 System.out.println(detectLoop(root));
		
		 System.out.println("~~~~************************");
		 System.out.println(printTopViewOfBT(root));
		 System.out.println(findLCA(root,
		 root.left.left,root.left.right.left).val);
		 System.out.println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
		 levelAverage(root);
		 System.out.println("~~~~*************************");
		
		 System.out.println("************************");
		 System.out.println(findKthTreeNode(root, 2).val);
		 System.out.println("*************************");
		
		 System.out.println("xxxxxxxxxxxxxxxxxxxxxxxxx");
		 printLevels(root,3,5);
		 System.out.println("xxxxxxxxxxxxxxxxxxxxxxxxx");
		
		 System.out.println("*************************");
		 printLevels(root,3,5);
		 System.out.println("*************************");
		
		 System.out.println("-------------");
		 System.out.println(printLeaf(root));
		 System.out.println(".................");
		 System.out.println(deepestLeaf(root).val + " ressssss");
		 System.out.println("-------------");
		 printRightView(root);
		 System.out.println();
		 System.out.println("-------------");
		 System.out.println("-------------");
		 printkdistanceNode(root, root.right.left, 4);
		 System.out.println();
		 System.out.println("-------------");
		 leftViewOfTree(root);
		 System.out.println();
		 getNoSiblingsNodes(root);
		 System.out.println();
		 System.out.println(isPairPresent(root, 18));
		 System.out.println(largestIndependentSet(root));
		 String seq = serializeTree(root);
		 System.out.println(seq);
		 String seq2 = serializeBTree(root);
		 System.out.println(seq2);
		 String seq3 = serializePreorder(root);
		 System.out.println(seq3);
		 String seq4 = serializePreorder0(root);
		 System.out.println(seq4);
		 TreeNode deserializedRoot=deserializeBTree(seq4);
		 inorder(deserializedRoot);
		 System.out.println();
		 System.out.println("xxxxxxxxxxxxxxxxxxxxYUAN");
		 // TreeNode r=deserializeTree(seq);
		 // inorder(r);
		 System.out.println();
		 System.out.println(distanceBetweenNodes(root, root.left.right.left,
		 root.left.left));
		 System.out.println(findDistance(root, 13, 3));
		
		 System.out.println("-------------");
		
		 System.out.println(kthLevelSum(root, 3));
		 System.out.println(longestPathLeafToLeaf(root));
		
		 int[][] matrix = { { 1, -2, -3, 4 }, { -2, 1, -3, -2 },
		 { 0, -2, -3, -1 }, { 4, 1, 6, -2 } };
		 System.out.println(findMaxFromFirstRow(matrix));
		
		 int[] nge = { 2, 6, 3, 4, 3, 1, 2 };
		 nextGreatestElement(nge);
		
		 System.out.println(findFirstNonRepeating("geeksforgeeksFirst"));
		 System.out.println(findFirstNonRepeating2("geeksforgeeksFirst"));
		 int[] num = { 1, 0, 1, 9 };
		 nextSmallestPalindrome(num);
		
		 String[] strs = { "yuan", "F", "feng", "eng", "eng", "f", "yuan",
		 "ff",
		 "s", "ff", "ff", "yuan", "ss", "F", "F", "F" };
		
		 topKStrings(strs, 4);
		 System.out.println();
		 int[] arr = { -1, 0, 0, 0, 0, -4 };
		 System.out.println(maxProductSubarray(arr));
		 int[] coins = { 1, 2, 5, 10 };
		 System.out.println(minMakeChange(coins, 12));
		
		 int[] pre = { 3, 2, 1, 4 };
		 TreeNode roo = constructBST(pre);
		 inorder(roo);
		
		 System.out.println();
		 char[][] board = { { 'X', 'O', 'X', 'O', 'X', 'O' },
		 { 'O', 'X', 'O', 'X', 'O', 'X' },
		 { 'X', 'O', 'X', 'O', 'X', 'O' },
		 { 'O', 'X', 'O', 'X', 'O', 'X' } };
		 solve(board);
		 solve2(board);
		 solve3(board);
		 solvex(board);
		 solvex1(board);
		
		 ListNode l1 = new ListNode(1);
		 ListNode l2 = new ListNode(2);
		 ListNode l3 = new ListNode(3);
		 ListNode l4 = new ListNode(4);
		 ListNode l5 = new ListNode(5);
		
		 l1.next = l2;
		 l2.next = l1;
		 l2.next = l3;
		 l3.next = l4;
		 l4.next = l5;
		 l5.next = l1;
		
		 ListNode newHead = deletNodeFromCirlularList(l1, 1);
		 ListNode cur = newHead;
		 do {
		 System.out.print(cur.val + " ");
		 cur = cur.next;
		 } while (cur != newHead);
		
		 System.out.println();
		
		 DListNode dl1 = new DListNode(1);
		 DListNode dl2 = new DListNode(2);
		 DListNode dl3 = new DListNode(3);
		 DListNode dl4 = new DListNode(4);
		 DListNode dl5 = new DListNode(5);
		
		 // dl1.next=dl1;
		 // dl1.pre=dl1;
		 dl1.next = dl2;
		 dl2.pre = dl1;
		 dl2.next = dl3;
		 dl3.pre = dl2;
		 dl3.next = dl4;
		 dl4.pre = dl3;
		 dl4.next = dl5;
		 dl5.pre = dl4;
		 dl5.next = dl1;
		 dl1.pre = dl5;
		
		 DListNode head1 = deletNodeFromDoublyCirlularList(dl1, 3);
		 DListNode curnode = head1;
		 do {
		 System.out.print(curnode.val + " ");
		 curnode = curnode.next;
		 } while (curnode != head1);
		 System.out.println();
		 System.out.println(findMin3DigitNum(100));
		 int[] Arr = { 1, 2, 100, 22, 28, 12 };
		 ArrayList<ArrayList<Integer>> result = subsetSum(Arr, 150);
		 for (int i = 0; i < result.size(); i++)
		 System.out.println(result.get(i));
		
		 System.out.println(reverseString("yuan fengpeng"));
		
		 TreeNodeP rootp = new TreeNodeP(5);
		 TreeNodeP n1 = new TreeNodeP(2);
		 TreeNodeP n2 = new TreeNodeP(8);
		 TreeNodeP n3 = new TreeNodeP(7);
		 TreeNodeP n4 = new TreeNodeP(6);
		 TreeNodeP n5 = new TreeNodeP(9);
		
		 TreeNodeP n6 = new TreeNodeP(4);
		 TreeNodeP n7 = new TreeNodeP(3);
		 TreeNodeP n8 = new TreeNodeP(1);
		
		 rootp.left = n1;
		 rootp.right = n2;
		 n2.left = n3;
		 n2.right = n5;
		 n3.left = n4;
		 n1.left = n8;
		 n1.right = n6;
		 n6.left = n7;
		
		 n1.parent = rootp;
		 n2.parent = rootp;
		 n3.parent = n2;
		 n4.parent = n3;
		 n5.parent = n2;
		 n6.parent = n1;
		 n7.parent = n6;
		 n8.parent = n1;
		
		 System.out.println(LCAncestor(n8, n5).val);
		 System.out.println(inorderSuccessorBST(n5));
		 System.out.println(inorderSucc(n6).val);
		 int[] nums = { 12, 11, 10, 5, 6, 2, 30 };
		 topKLargestestNum(nums, 4);
		 System.out.println();
		 System.out.println(Arrays.toString(nums));
		 Arrays.sort(nums);
		 System.out.println(Arrays.toString(nums));
		
		 inorderTrav(root);
		
		 int[][] maze = { { 0, 0, 0, 0, 1, 1, 1 }, { 1, 1, 1, 0, 1, 1, 1 },
		 { 0, 0, 0, 0, 1, 1, 1 }, { 0, 1, 1, 1, 0, 0, 0 },
		 { 0, 0, 1, 0, 0, 1, 0 }, { 1, 0, 1, 0, 1, 1, 0 },
		 { 1, 0, 0, 0, 1, 1, 0 } };
		 // ArrayList<Pair> res=findPathOfMaze(maze,0,0);
		 // for(int i=0;i<res.size();i++){
		 // System.out.println(res.get(i));
		 // }
		 System.out.println(totalNodesOfLevelK(root, 4));
		
		 int[][] matrix1 = { { 1, 5, 10, 12 }, { 3, 8, 11, 13 },
		 { 7, 9, 14, 16 }, { 20, 21, 22, 23 } };
		 System.out.println(findKthOfSortedMatrix(matrix1, 12));
		 System.out.println(kthLargestOfSortedMatrix(matrix1, 12));
		
		 int cost[][] = { { 5, 8, 2, 6 }, { 1, 3, 8, 2 }, { 6, 6, 5, 2 },
		 { 4, 6, 1, 6 } };
		 System.out.println(findPathWithMaxSum(cost));
		 int arry[] = { 4, 5, 6, 7, 8, 9, 1, 2, 3 };
		 System.out.println(findMinOfRotatedArray(arry));
		 int[] arrry = { 3, 2, 4, 1, 2, 3, 4, 1, 5, 4, 7, 3, 8 };
		 quickSort(arrry);
		
		 int[] twosum = { 1, 4, 45, 6, 8 };
		 System.out.println(hasArrayTwoCandidates(twosum, 16));
		
		 ListNode h = new ListNode(1);
		 ListNode h1 = new ListNode(2);
		 ListNode h2 = new ListNode(3);
		 ListNode h3 = new ListNode(4);
		 ListNode h4 = new ListNode(5);
		 ListNode h5 = new ListNode(6);
		 h.next = h1;
		 h1.next = h2;
		 h2.next = h3;
		 h3.next = h4;
		 h4.next = h5;
		
		 // ListNode rh=reverseListRecur(h);
		 ListNode rh = reverseListIterative(h);
		 while (rh != null) {
		 System.out.print(rh.val + " ");
		 rh = rh.next;
		 }
		 System.out.println();
		
		 System.out.println(anagram("ayuanw", "anyuwa"));
		 System.out.println(anagram2("ayuanw", "anyuwa"));
		 mergeSort(arrry);
		 int[] prices1 = { 7, 6, 5, 4, 3, 2, 1 };
		 System.out.println(maxProfit(prices1));
		 char[] ch = { 'a', 'c', 'f', 'j', 'k' };
		 System.out.println(findNextChar(ch, 'b'));
		
		 // char[] ch1={'c', 'f', 'j', 'p', 'v'};
		 // System.out.println(findNextChar(ch1,'a'));
		 //
		 // char[] ch2={'c', 'f', 'j', 'p', 'v'};
		 // System.out.println(findNextChar(ch2,'c'));
		 // char[] ch3={'c', 'f', 'j', 'p', 'v'};
		 // System.out.println(findNextChar(ch3,'k'));
		
		 System.out.println(findDistanceBetweenWords("hello how are hello",
		 "hello", "you"));
		
		 System.out.println(findDepth("((00)(0(00)))"));
		
		 int[] arrsum = { 1, 3, 5 };
		 System.out.println(arraySum(arrsum));
		
		 TreeNode root1 = new TreeNode(10);
		 root1.left = new TreeNode(5);
		 root1.right = new TreeNode(100);
		 root1.right.left = new TreeNode(50);
		 root1.right.right = new TreeNode(150);
		 root1.right.left.left = new TreeNode(40);
		
		 System.out.println(inorderSuccessor(root1, root1.right.left).val);
		 System.out.println("****************************OOOOOOOOOOOO");
		 System.out.println(isBalanced(root1));
		 String testS = "   a   b ";
		 System.out.println(reverseWords2(testS));
		
		 int arr3[] = { 2, 1, 3, 4 };
		 System.out.println(findGroupsMultipleOf3(arr3));
		
		 System.out.println(match("*pqrs", "pqrst"));
		
		 System.out.println(name);
		 changeName("yuan");
		 System.out.println(name);
		
		 System.out.println(lcs("12345yu", "13579uy"));
		 int arr1[] = { 1, 10, 5, 2, 7 };
		 int x = 9;
		
		 System.out.println(smallestSubWithSum(arr1, x));
		 System.out.println(smallestSubWithSum2(arr1, x));
		
		 int[] removal = { 20, 4, 1, 3 };
		 System.out.println(minRemovals(removal));
		 System.out.println(minRemovals2(removal));
		
		 generateMatrix0X(5, 4);
		 System.out.println();
		 generateMatrix0X2(5, 4);
		
		 System.out.println(simplifyPath("/.."));
		
		 ListNode duphead = new ListNode(1);
		 ListNode dups = new ListNode(1);
		 ListNode nd1 = new ListNode(2);
		 ListNode nd2 = new ListNode(2);
		 ListNode nd3 = new ListNode(4);
		 ListNode nd4 = new ListNode(4);
		 ListNode nd5 = new ListNode(5);
		
		 duphead.next = dups;
		 // dups.next=nd1;
		 // nd1.next=nd2;
		 // nd2.next=nd3;
		 // nd3.next=nd4;
		 // nd4.next=nd5;
		
		 ListNode resNode = deleteDuplicatesAll(duphead);
		 while (resNode != null) {
		 System.out.print(resNode.val + " ");
		 resNode = resNode.next;
		 }
		 ArrayList<Integer> lst1 = new ArrayList<Integer>();
		 lst1.add(1);
		 lst1.add(2);
		 lst1.add(3);
		 lst1.add(4);
		 lst1.add(5);
		
		 ArrayList<Integer> lst2 = new ArrayList<Integer>();
		 lst2.add(2);
		 lst2.add(4);
		 lst2.add(6);
		 lst2.add(8);
		 lst2.add(10);
		
		 System.out.println(intersection(lst1, lst2));
		
		 System.out.println(getPermutation(1, 1));
		
		 int[] arrS={1,1,1,5};
		 System.out.println(findSmallest(arrS));
		
		
		 //Test No 1
		 RandomTreeNode tree = new RandomTreeNode(1);
		 tree.left = new RandomTreeNode(2);
		 tree.right = new RandomTreeNode(3);
		 tree.left.left = new RandomTreeNode(4);
		 tree.left.right = new RandomTreeNode(5);
		 tree.random = tree.left.right;
		 tree.left.left.random = tree;
		 tree.left.right.random = tree.right;
		
		 printInorder(tree);
		
		 int graph[][] = { {0, 1, 1, 1},
		 {0, 0, 0, 1},
		 {0, 0, 0, 1},
		 {0, 0, 0, 0}
		 };
		 int u = 0, v = 3, k = 2;
		 System.out.println(countwalks(graph, u, v, k));
		 System.out.println(countwalksDp(graph, u, v, k));
		
		 System.out.println(countDecoding("121"));
		 System.out.println(countDecodingDp("121"));
		
		 int graph1[][] = { {0, 10, 3, 2},
		 {Integer.MAX_VALUE, 0, Integer.MAX_VALUE, 7},
		 {Integer.MAX_VALUE, Integer.MAX_VALUE, 0, 6},
		 {Integer.MAX_VALUE, Integer.MAX_VALUE, Integer.MAX_VALUE, 0}
		 };
		 System.out.println(shortestPathRecur(graph1, 0, 3,2));
		 System.out.println(shortestPathDp(graph1, 0, 3,2));
		
		 int ropes[] = {4, 3, 2, 6};
		 System.out.println(minCost(ropes));
		
		 int array[] = {3, 2, 10, 4, 40};
		 System.out.println(binarySearch(array, 4));
		
		 TreeNode rooot=new TreeNode(1);
		 rooot.left=new TreeNode(2);
		 rooot.right=new TreeNode(3);
		 rooot.left.left=new TreeNode(4);
		 rooot.left.right=new TreeNode(5);
		 rooot.left.right.right = new TreeNode(15);
		 rooot.right.left = new TreeNode(6);
		 rooot.right.right = new TreeNode(7);
		 rooot.right.left.right = new TreeNode(8);
		
		 TreeNode Node1 = rooot.left.left;
		 TreeNode Node2 = rooot.right.left;
		 System.out.println(getLevel(rooot,Node1));
		 System.out.println(getLevel(rooot,Node2));
		 System.out.println(isSibling(rooot,Node1, Node2));
		 System.out.println(isCousin(rooot, rooot.left.left,
		 rooot.right.right));
		
		 int ar1[] = {1, 5, 10, 20, 40, 80};
		 int ar2[] = {6, 7, 20, 80, 100};
		 int ar3[] = {3, 4, 15, 20, 30, 70, 80, 120};
		 findCommon(ar1,ar2,ar3);
		
		 System.out.println(isIsomorphic("foo","app"));
		 System.out.println(isIsomorphic("bar","foo"));
		 System.out.println(isIsomorphic("turtle", "tletur"));
		 System.out.println(isIsomorphic("ab","ca"));
		
		 String word="hello how are hello";
		 String[] words=word.split(" ");
		 System.out.println(minDistStrings(words,"hello","you"));
		 System.out.println(minDistStringsRelative(words,"hello","you"));
		
		 TreeNode flipRoot=new TreeNode(1);
		 flipRoot.left=new TreeNode(2);
		 flipRoot.right=new TreeNode(3);
		 flipRoot.left.left=new TreeNode(4);
		 flipRoot.left.right=new TreeNode(5);
		 flipRoot.left.left.left=new TreeNode(6);
		 flipRoot.left.left.right=new TreeNode(7);
		 inorder(flipRoot);
		 System.out.println();
		 TreeNode flipDown=flipDown(flipRoot);
		 inorder(flipDown);
		 System.out.println();
		
		 int[][] bipartite={{0, 1, 0, 1},
		 {1, 0, 1, 0},
		 {0, 1, 0, 1},
		 {1, 0, 1, 0}
		 };
		
		 System.out.println(isBipartite(bipartite,0));
		
		 int parent[] = {-1, 0, 0, 1, 1, 3, 5};
		 System.out.println(findHeight(parent));
		
		 System.out.println(printFactors(24));
		
		 ComplexNode chead=new ComplexNode(10);
		 chead.next=new ComplexNode(5);
		 chead.next.next=new ComplexNode(12);
		 chead.next.next.next=new ComplexNode(7);
		 chead.next.next.next.next=new ComplexNode(11);
		 chead.child=new ComplexNode(4);
		 chead.child.next=new ComplexNode(20);
		 chead.child.next.next=new ComplexNode(13);
		 chead.child.next.child=new ComplexNode(2);
		 chead.child.next.next.child=new ComplexNode(16);
		 chead.child.next.next.child.child=new ComplexNode(3);
		 chead.next.next.next.child=new ComplexNode(17);
		 chead.next.next.next.child.next=new ComplexNode(6);
		 chead.next.next.next.child.child=new ComplexNode(9);
		 chead.next.next.next.child.child.next=new ComplexNode(8);
		 chead.next.next.next.child.child.child=new ComplexNode(19);
		 chead.next.next.next.child.child.child.next=new ComplexNode(15);
		 ComplexNode resCNode=flattenList(chead);
		 while(resCNode!=null){
		 System.out.print(resCNode.val+" ");
		 resCNode=resCNode.next;
		 }
		 System.out.println();
		
		 System.out.println(removeDuplicates("geeksforgeeks"));
		 int[] A1={1, 2, 3, 4, 5, 10, 9, 8, 7, 6};
		 System.out.println(TurningNumberIndex(A1));
		 int[] A2={-3, -1, 1, 3, 5};
		 System.out.println(getNumberSameAsIndex(A2));
		
		 int[] A3={1,2,3};
		 System.out.println(findMinimum(A3));
		
		 System.out.println(printSeq(4,2));
		
		 int mat[][] = { {10, 20, 30, 40},
		 {15, 25, 35, 45},
		 {27, 29, 37, 48},
		 {32, 33, 39, 50},
		 };
		 System.out.println(printSorted(mat));
		
		 int preorder[] = {10, 5, 1, 7, 40, 50};;
		
		 inorder(constructBST(preorder));
		
		 System.out.println();
		
		 int M[][]= { {1, 1, 0, 0, 0},
		 {0, 1, 0, 0, 1},
		 {1, 0, 0, 1, 1},
		 {0, 0, 0, 0, 0},
		 {1, 0, 1, 0, 1}
		 };
		
		 int[][] island={{1, 0, 1, 0},
		 {0, 1, 0, 1},
		 {1, 1, 0, 0},
		 {0, 0, 0, 1}
		
		 };
		
		 int[][] is1={{1, 1, 1},
		 {1, 0, 1},
		 {0, 1, 0}};
		 System.out.println(numOfIslands(M));
		 System.out.println(numOfIslands(island));
		
		 System.out.println(numOfIslands(is1));
		 System.out.println(areaOfIsland(is1,0,0));
		 System.out.println(areaOfIsland2(is1,0,0));
		
		 System.out.println(longestCommonString("OldSite:GeeksforGeeks.org","NewSite:GeeksQuiz.com"));
		 System.out.println(oneEditApart("cat","act"));
		
		 TreeNode bst1=new TreeNode(7);
		 bst1.left=new TreeNode(3);
		 bst1.right=new TreeNode(9);
		 bst1.left.right=new TreeNode(6);
		 bst1.right.right=new TreeNode(11);
		
		 TreeNode bst2=new TreeNode(9);
		 bst2.left=new TreeNode(7);
		 bst2.right=new TreeNode(13);
		 bst2.left.left=new TreeNode(6);
		 bst2.right.left=new TreeNode(11);
		
		 intersectionOfTwoBST(bst1,bst2);
		 System.out.println();
		
		 int arrClosest[] ={12, 16, 22, 30, 35, 39, 42,
		 45, 48, 50, 53, 55, 56};
		 findKClosest(arrClosest,35,4);
		 System.out.println();
		 System.out.println(countPalindromeDP1("abbab"));
		 System.out.println(countPalindromeDP2("abbab"));
		 System.out.println(countPalindrome2("abbab"));
		
		 int[] zeroArr={0,0,0,0,3,4,1};
		 reArrange2(zeroArr);
		 System.out.println(Arrays.toString(zeroArr));
		
		 int[] powerset={1,2,3};
		 System.out.println(powerSet(powerset));
		
		 System.out.println(sqrt(5.0));
		
		 int[][] AT={{1,1,2},
		 {0,3,1},
		 {2,1,0}};
		 int[][] B1=transformMatrix(AT);
		 for(int i=0;i<B1.length;i++)
		 System.out.println(Arrays.toString(B1[i]));
		 System.out.println();
		 int[][] B2=transformMatrix2(AT);
		 for(int i=0;i<B2.length;i++)
		 System.out.println(Arrays.toString(B2[i]));
		 System.out.println();
		
		 ListNode ln1=new ListNode(1);
		 ln1.next=new ListNode(2);
		 ln1.next.next=new ListNode(3);
		 ln1.next.next.next=new ListNode(4);
		
		 ListNode ln2=new ListNode(5);
		 ln2.next=new ListNode(6);
		
		 // ListNode nh=interleave(ln1,ln2);
		 // while(nh!=null){
		 // System.out.print(nh.val+" ");
		 // nh=nh.next;
		 // }
		 // System.out.println();
		
		 ListNode nh2=interleave2(ln1,ln2);
		 while(nh2!=null){
		 System.out.print(nh2.val+" ");
		 nh2=nh2.next;
		 }
		 System.out.println();
		
		 int set1[] = {1, 7, 10, 15, 27, 29};
		 int set2[] = {5, 10, 15, 20, 25, 30};
		 System.out.println(lengthLongestAP(set1)+" "+lengthLongestAP(set2));
		
		 Job[] jobs = { new Job('a', 2, 100), new Job('b', 1, 19), new
		 Job('c', 2, 27),
		 new Job('d', 1, 25), new Job('e', 3, 15)};
		 jobScheduling(jobs);
		
		 int arr5[] = {1,2,3,11,7,2,5,6};
		 System.out.println(duplicateWithinkIndices(arr5,3));
		
		 int[] AA1={1,2,2};
		 int[]BB1={2,0,1,1};
		 System.out.println(adjust(AA1,BB1));
		
		 int[] window={1, 3, -1, -3, 5, 3, 6, 7};
		 System.out.println(maxSlidingWindow(window,3));
		 System.out.println(Arrays.toString(minSlidingWindow(window,3)));
		 System.out.println(Arrays.toString(minSlidingWindow2(window,3)));
		
		 int[] coins1={8, 15, 3, 7};
		 int[] coins2={2, 2, 2, 2};
		 int[] coins3={20, 30, 2, 2, 2, 10};
		 System.out.println(coins(coins1)+" "+coins(coins2)+" "+coins(coins3));
		
		 System.out.println(maxString("452","178"));
		
		 System.out.println(longestPalindromeDP("bb"));
		
		 int[] nums1={1,2};
		 System.out.println(findPeakElement(nums1));
		
		 int[] A7={5,5,2,3,4,7,5};
		 System.out.println(findIndex(A7,4));
		 System.out.println(numDecodingsConst("26"));
		 System.out.println(minWindow3("ADOBECODEBANC","ABC"));
		
		 int[] ratings={4,2,3,4,1};
		 System.out.println(candy(ratings));
		
		 System.out.println(letterCombinationsIterative("234"));
		
		
		 System.out.println(sol.letterCombinations("234"));
		
		 System.out.println(sol.generateParenthesisIterative(3));
		
		 System.out.println(sol.isScrambleRecur("ccbbcaccbccbbbcca",
		 "ccbbcbbaabcccbccc"));
		
		 int[] positives={1,2,4,5,6};
		 System.out.println(sol.firstMissingPositive2(positives));
		 System.out.println(sol.compareVersion("1", "0"));
		
		 System.out.println(sol.fractionToDecimal(-1, -2147483648));
		
		 // int val[] = {60, 100, 120};
		 // int wt[] = {10, 20, 30};
		 // int W = 50;
		 int val[] = {1,4,2,3};
		 int wt[] = {2,1,4,3};
		 int W = 9;
		
		 System.out.println(sol.knapsack(W, val, wt));
		 System.out.println(sol.knapsackReduceSpace(W, val, wt));
		
		 int[] water={2,0,2};
		 System.out.println(sol.trap2(water));
		 System.out.println(sol.trap3(water));
		 System.out.println(sol.pow(34.00515, -3));
		
		 sol.fill(7);
		
		 System.out.println(sol.sqrtBinary(4));
		
		 int[] random={1,2,3,4,5,6,7,8,9,10,11,12,13,1,4,14};
		 System.out.println(Arrays.toString(sol.randomSelectK(random, 4)));
		
		 int[] test7={3,4,5,8,8,8,8,10,13,14};
		 System.out.println(sol.binarySearch2(test7, 8));
		
		 ArrayList<Integer> elements=new ArrayList<Integer>();
		 elements.add(9);
		 elements.add(3);
		 elements.add(2);
		 elements.add(4);
		 elements.add(8);
		
		 System.out.println(sol.kthLargestElement(3, elements));
		
		 int[] eles={1,2,3,4,5,6,8,9,10,7};
		 System.out.println(sol.findKthLargest(eles, 0, 9, 10));
		
		 System.out.println(sol.binarySQRT(1));
		
		 ListNode d1=new ListNode(3);
		 // ListNode d2=new ListNode(9);
		 // ListNode d3=new ListNode(12);
		 // ListNode d4=new ListNode(15);
		 // ListNode d5=new ListNode(21);
		 // d1.next=d2;
		 // d2.next=d3;
		 // d3.next=d4;
		 // d4.next=d5;
		
		 ListNode dres=sol.deleteEveryOther(d1);
		
		 while(dres!=null){
		 System.out.print(dres.val+" ");
		 dres=dres.next;
		 }
		 System.out.println();
		
		
		 ListNode cl11=new ListNode(1);
		 ListNode cl12=new ListNode(2);
		 ListNode cl13=new ListNode(3);
		 ListNode cl14=new ListNode(5);
		 ListNode cl15=new ListNode(6);
		
		 cl11.next=cl12;
		 cl12.next=cl13;
		 cl13.next=cl14;
		 cl14.next=cl15;
		
		 ListNode cl21=new ListNode(2);
		 ListNode cl22=new ListNode(3);
		 ListNode cl23=new ListNode(5);
		 ListNode cl24=new ListNode(6);
		 // ListNode cl25=new ListNode(8);
		
		 cl21.next=cl22;
		 cl22.next=cl23;
		 cl23.next=cl24;
		 // cl24.next=cl25;
		
		 ListNode comm=sol.commonNodes(cl11,cl21);
		 while(comm!=null){
		 System.out.print(comm.val+" ");
		 comm=comm.next;
		 }
		 System.out.println();
		
		 ListNode lh=new ListNode(1);
		 lh.next=new ListNode(2);
		 lh.next.next=new ListNode(3);
		
		 ListNode rsh=sol.addBefore(lh, 4, 3);
		 while(rsh!=null){
		 System.out.print(rsh.val+" ");
		 rsh=rsh.next;
		 }
		
		
		 int[] nums12={-1,-100,3,99};
		 sol.rotate(nums12, 3);
		 System.out.println(Arrays.toString(nums12));
		
		 System.out.println(sol.replaceString("a?b?c?"));
		
		 int[] t1={1,2};
		 System.out.println(sol.permuteUnique(t1));
		
		 System.out.println(sol.generate_AllPalindromes("carecra"));
		
		 System.out.println(sol.reverseBits(43261596));
		
		 System.out.println(sol.isWellFormed("(hkjkj(kk)[kkl]aabb78]",
		 "ab78"));
		 System.out.println(sol.hammingWeight(11));
		
		 System.out.println(sol.square(8));
		
		 System.out.println(sol.findOptimal(10));
		 char islandmat[][] = {{'X', 'O', 'O', 'O', 'O', 'O'},
		 {'X', 'O', 'X', 'X', 'X', 'X'},
		 {'O', 'O', 'O', 'O', 'O', 'O'},
		 {'X', 'X', 'X', 'O', 'X', 'X'},
		 {'X', 'X', 'X', 'O', 'X', 'X'},
		 {'O', 'O', 'O', 'O', 'X', 'X'},
		 };
		
		 System.out.println(countIslands(islandmat));
		
		 System.out.println(sol.maxDepParenthesis("( ((X)) (((Y))) )"));
		 int AR1[] = {3, 5, 7, 9, 10, 90, 100, 130, 140, 160, 170};
		 System.out.println(sol.findPos(AR1, 120));
		
		 System.out.println(sol.countWays(20));
		 System.out.println(sol.getPeriod(6));
		
		 System.out.println(sol.getPeriod2(6));
		
		 System.out.println(sol.findLength("123123"));
		 System.out.println(sol.findLength2("123123"));
		
		 System.out.println(sol.kUniques("aabacbebebe", 2));
		 System.out.println(sol.kUniques("aabaadddddaa", 2));
		 int[] prices4 = { 1,4,3,6,5,2, 4};
		 System.out.println(sol.maxProfit4(3, prices4));
		 System.out.println(sol.atoi2("1.0"));
		
		 HashSet<String> dic=new HashSet<String>();
		 dic.add("a");
		 dic.add("b");
		 dic.add("c");
		
		 System.out.println(sol.findLadders2("a", "c", dic));
		 int[] ar={4, 6, 9, 2, 3, 4, 9, 6, 10, 4};
		 System.out.println(Arrays.toString(sol.orderedGroup(ar)));
		
		 System.out.println(sol.buildSmallestNumber("765028321", 5));
		 System.out.println(sol.isMatch2("abcd", "?*c"));
		 String path="GLLG";
		 System.out.println(sol.isCircular(path));
		
		 int[] rotated = {11, 15, 6, 8, 9, 10};
		 System.out.println(sol.twoSumRotated(rotated, 13));
		 int[] wiggle={2, 3, 5, 6, 7};
		 sol.wiggleSort(wiggle);
		 System.out.println(Arrays.toString(wiggle));
		
		 int[] range={0, 1, 3, 50, 75, 100};
		 System.out.println(sol.findMissingRanges(range, 0, 80));
		
		 System.out.println(sol.lengthOfLongestSubstringTwoDistinct("eceabab"));
		 System.out.println(sol.lengthOfLongestSubstringTwoDistinct2("eceabab"));
		 System.out.println(sol.lengthOfLongestSubstringKDistinct("eceabab",2));
		
		 List<Integer> tasks=new ArrayList<Integer>();
		 tasks.add(1);
		
		 tasks.add(1);
		 tasks.add(2);
		 tasks.add(2);
		
		 System.out.println(sol.rescheduleTask(tasks, 3));
		 System.out.println(sol.distinctSubsequences("waeginsapnaabangpisebbasepgnccccapisdnfngaabndlrjngeuiogbbegbuoecccc",
		 "a+b+c-"));
		 // System.out.println(sol.distinceSubsequences("waeginsapnccccaab",
//		 "c-"));
		 System.out.println(sol.distinctSubsequencesRecur("waeginsapnaabangpisebbasepgnccccapisdnfngaabndlrjngeuiogbbegbuoecccc",
		 "a+b+c-"));
		 //
		 System.out.println(sol.distinctSubsequences2("waeginsapnaabangpisebbasepgnccccapisdnfngaabndlrjngeuiogbbegbuoecccc",
		 "a+b+c-"));
		 //
		 System.out.println(sol.distinctSubsequencesRecur("waaeginsaapnccccaab",
		 "a+c-"));
		 System.out.println(sol.distinctSubsequences2("waaeginsaapnccccaab",
		 "a+c-"));
		 System.out.println("---------------");
		 int[][] policefinder1={{0, 1, 0},
		 {0, 2, 0},
		 {0, 1, 0}};
		 int[][] policefinder2={{0, 2, 0},
		 {0, 2, 0},
		 {0, 2, 1}};
		 int[][] policefinder3={{1, 0, 0},
		 {0, 0, 0},
		 {0, 0, 1}};
		 sol.findNearestPolice(policefinder1);
		
		 sol.findNearestPolice(policefinder2);
		 sol.findNearestPolice(policefinder3);
		 int[][] police={{1, 2,1},
		 {1, 0, 1},
		 {1, 2, 1}};
		 sol.findNearestPolice2(police);
		
		 int[] bits={1,0,0,1,0, 0, 1};
		 System.out.println(sol.flipBits(bits));
		 // System.out.println(sol.bitFlip(bits));
		 System.out.println("-------*********-------*********");
		 int[] testcase1={1, 5, 3, 4, 2};
		 int[] testcase2={10, 363374326, 364147530, 61825163, 1073065718,
		 1281246024, 1399469912, 428047635, 491595254, 879792181, 1069262793};
		 System.out.println(sol.totalPairOfDiffK(testcase1, 2));
		 System.out.println(sol.totalPairOfDiffK(testcase2, 1));
		
		 char[][]
		 cavity={{'1','1','1','2'},{'1','9','1','2'},{'1','8','9','2'},{'1','2','3','4'}};
		 sol.findCavity(cavity);
		
		 int missing[] = {88, 105, 3, 2, 200, 0, 10};
		 sol.findMissingRanges(missing, 100);
		
		 TreeNode r=new TreeNode(9);
		 r.left=new TreeNode(10);
		 r.right=new TreeNode(12);
		 r.left.left=new TreeNode(8);
		 r.left.right=new TreeNode(5);
		 r.right.right=new TreeNode(15);
		
		 System.out.println(sol.printPathBy5(r));
		
		 int[] before={1,2,3,4,5};
		 int[] after={3,4,5,2,1};
		
		 System.out.println(sol.pushPopSequence(before, after));
		
		 int repeateArr[] = {1, 2, 3, 1, 3, 6, 6, 1};
		 sol.findRepettion(repeateArr);
		
		 int[] preorder1={3,1,4,2,5};
		 System.out.println(sol.isValidPreorder(preorder1));
		
		 int[][] ocean={{1,0,0,1},{1,0,0,1},{1,1,0,0}};
		 int[][] ocean2={{1,1,1,0},{1,0,1,0},{1,1,1,0}};
		 System.out.println(sol.findIslandSize(ocean2));
		
		 // Scanner in = new Scanner(System.in);
		 // int T=in.nextInt();
		 //
		 // for(int i=0;i<T;i++){
		 // int N=in.nextInt();
		 // int shares[]=new int[N];
		 // for(int c=0;c<N;c++){
		 // shares[c]=in.nextInt();
		 // }
		 // System.out.println(sol.buySellStocks(shares));
		 // }
		 //
		 // in.close();
		
		 List<Integer> mapping =new ArrayList<Integer>();
		 mapping.add(2);
		 mapping.add(3);
		 mapping.add(1);
		 mapping.add(2);
		 mapping.add(2);
		 // mapping.add(6);
		 // System.out.println(sol.transform2GoodNodes(mapping));
		
		 System.out.println(sol.paintWays(3));
		 System.out.println(sol.paintWays2(3));
		
		 String[]
		 strs01={"1","1110","10001","01","001","1001","101","10","00"};
		 System.out.println(sol.longest01Strings(strs01, 1, 2));
		
		 int[] A100={1,1,1,1};
		 System.out.println(sol.maxValueOfArray(A100));
		 System.out.println(sol.maxValueOfArray2(A100));
		 System.out.println();
		 System.out.println(sol.isHappy(2));
		
		 Pair pair1=new Pair("Alex", "Boss");
		 Pair pair2=new Pair("Bob","Charlie");
		 Pair pair3=new Pair("Charlie", "Greg");
		 Pair pair4=new Pair("Tom", "Jerry");
		 Pair pair5=new Pair("Greg", "Hollister");
		
		 List<Pair> friends=new ArrayList<Pair>();
		 friends.add(pair1);
		 friends.add(pair2);
		 friends.add(pair3);
		 friends.add(pair4);
		 friends.add(pair5);
		
		 System.out.println(sol.findFriends(friends));
		
		 int[][] sortedArrays={{2,4,6,8,9,12,14,16},
		 {3,6,7,9,22,25,28},
		 {2,5,7,8,10,11,16},
		 {4,8,23,26,28}};
		 sol.mergeNSortedArrays(sortedArrays);
		
		 int[][] flows={{1, 2, 2, 3, 5},
		 {3, 2, 3, 4, 4},
		 {2, 4, 5, 3, 1 },
		 {6, 7, 1, 4, 5 },
		 {5, 1, 1, 2, 4}};
		
		 System.out.println(sol.flowing_water(flows));
		
		 int[] medians={3,7,2,4,1,9};
		 System.out.println(sol.medianSlidingWindow(medians, 2));
		 // System.out.println(sol.medianSlidingWindow2(medians, 3));
		
		 System.out.println(sol.shuffleWords("I want to get a cup of water"));
		 System.out.println(sol.numOfAnagramSubstrings("acbacb","abc"));
		 System.out.println(sol.countStr("acbacb","abc"));
		 System.out.println();
		 // String[] expression={"[", "+", "1", "2", "4","5","[","*", "1","2",
//		 "3","4","]","]"};
		 // String[] expression1={"+", "1", "2", "3"};
		 // String[] expression2={"+"};
		 // String[] expression3={"*"};
		 // String[] expression4={"/","6","2","3"};
		 // String[] expression5={"100"};
		 // String[] expression6={"[", "+", "1", "2", "3", "[","/","[","*",
//		 "2", "3","]", "6","]","]"};
		 // String[] expression7={"[","+", "1", "2", "3","]"};
		 // System.out.println(sol.evaluateExpression(expression));
		 // System.out.println(sol.evaluateExpression(expression1));
		 // System.out.println(sol.evaluateExpression(expression2));
		 // System.out.println(sol.evaluateExpression(expression3));
		 // System.out.println(sol.evaluateExpression(expression4));
		 // System.out.println(sol.evaluateExpression(expression5));
		 // System.out.println(sol.evaluateExpression(expression6));
		 // System.out.println(sol.evaluateExpression(expression7));
		 int[] tickets={2,5, 6, 7};
		 System.out.println(sol.sellTicket(tickets, 4));
		
		 int triple[] = {3, 1, 4, 6, 5};
		 System.out.println(sol.isTriplet(triple));
		
		 TreeNode roooot=new TreeNode(1);
		 System.out.println(sol.countNodes1(roooot));
		 System.out.println(sol.combinationSum3(3, 7));
		 System.out.println("............");
		 System.out.println(sol.calculate("1 + 1"));
		
		 int[] nums100={0,0,0};
		 System.out.println(sol.majorityElement(nums100));
		
		 int[] A101 = {-7, 1, 5, 2, -4, 3, 0};
		 System.out.println(sol.equilibrium(A101));
		
		 ListNode lstTestHead=new ListNode(1);
		 lstTestHead.next=new ListNode(2);
		 lstTestHead.next.next=new ListNode(3);
		 lstTestHead.next.next.next=new ListNode(4);
		
		 int matt[][] = {{1, 1, 2}, {3, 4, 6}, {5, 3, 2}};
		 Cell[] cells= {new Cell(0, 0), new Cell(1, 1), new Cell(0, 1)};
		
		 sol.printSum(matt, cells);

		System.out.println("sssss~~~");
		System.out.println(sol.evaluateExpression("3456237490",1185));
		System.out.println(sol.evaluateExpression("1231231234", 11353));
		System.out.println(sol.calculation("1231231234", 11353));
		
		System.out.println(sol.reverseString2("@@!this,,,is.a@ word!am"));
		
		int[] t={};
		sol.maxSlidingWindow2(t, 0);
		
		char[][] brd={"bbaaba".toCharArray(),"bbabaa".toCharArray(),"bbbbbb".toCharArray(),"aaabaa".toCharArray(),"abaabb".toCharArray()};
		System.out.println(sol.findWords(brd,new String[]{"abbbababaa"}));
		
		System.out.println(sol.calculate2("3 + 2 * 2 - 9 / 3"));
		
		String[] words_test = {"practice", "makes", "perfect", "coding", "makes"};
		System.out.println(sol.shortestDistance(words_test, "makes", "coding"));
		
		System.out.println(sol.generateParenthesisDP(3));
		
		int[] nns={1,2,2};
		System.out.println(sol.subsetsWithDup2(nns));
	}

}
