package linkedIn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Random;
import java.util.Set;
import java.util.Stack;
import java.util.StringTokenizer;
import java.util.TreeSet;

public class Solutions {

	public static String serializeBinaryTree(TreeNode root) {
		StringBuilder sb = new StringBuilder();
		serializeBinaryTreeUtil(root, sb);
		return sb.toString();
	}

	public static void serializeBinaryTreeUtil(TreeNode root, StringBuilder sb) {
		if (root == null) {
			sb.append("# ");
			return;
		}
		sb.append(root.val + " ");
		serializeBinaryTreeUtil(root.left, sb);
		serializeBinaryTreeUtil(root.right, sb);
	}

	public static TreeNode deserialize(String s) {
		if (s == null || s.length() == 0)
			return null;
		StringTokenizer st = new StringTokenizer(s, " ");
		return deserializeUtil(st);
	}

	public static TreeNode deserializeUtil(StringTokenizer st) {
		if (!st.hasMoreTokens())
			return null;
		String val = st.nextToken();
		if (val.equals("#"))
			return null;
		TreeNode root = new TreeNode(Integer.parseInt(val));
		root.left = deserializeUtil(st);
		root.right = deserializeUtil(st);
		return root;
	}

	public static void inorder(TreeNode root) {
		if (root == null)
			return;
		inorder(root.left);
		System.out.print(root.val + " ");
		inorder(root.right);
	}

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

	public interface N {
		boolean isInteger();

		Integer getInteger();

		List<N> getList();
	}

	public static int leveSum(List<N> list) {
		if (list.size() == 0)
			return 0;
		return levelSumUtil(list, 1);
	}

	public static int levelSumUtil(List<N> list, int level) {
		int sum = 0;
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i).isInteger())
				sum += list.get(i).getInteger() * level;
			else
				sum += levelSumUtil(list.get(i).getList(), level + 1);
		}
		return sum;
	}

	public static int levelSum2(List<Object> list) {
		if (list.size() == 0)
			return 0;
		return levelSumUtil2(list, 1);
	}

	public static int levelSumUtil2(List<Object> list, int level) {
		int sum = 0;
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i) instanceof Integer)
				sum += (int) list.get(i) * level;
			else
				sum += levelSumUtil2((List<Object>) list.get(i), level + 1);
		}
		return sum;
	}
	
	
	public static int getLevel(List<Object> list){
		if(list.size()==0)
			return 0;
		int dep=1;
		for(int i =0;i<list.size();i++){
			if (list.get(i) instanceof List)
				dep = Math.max(dep, 1+getLevel((List<Object>) list.get(i)));
		}
		return dep;
	}
	
	public static int reverseLevelSum(List<Object> list){
		int dep = getLevel(list);
		int[] sum={0};
		reverseLevelSumUtil(list, dep, 1, sum);
		return sum[0];
	}
	// reverse sum level linkedin
	public static void reverseLevelSumUtil(List<Object> list, int dep, int level, int[] sum){
		for(int i=0;i<list.size();i++){
			if (list.get(i) instanceof List){
				reverseLevelSumUtil((List<Object>)list.get(i), dep, level+1, sum);
			}
			else
				sum[0] += (int)list.get(i)*(dep-level+1);
		}
	}
	
	
//	def revSum(nlist):
//	    (depth, res) = dfs(nlist)
//	    return res
//
//	def dfs(nlist):
//	    nlist_sum = 0
//	    s = 0
//	    max_lv = 1
//	    for item in nlist:
//	        if isinstance(item, list):
//	            (depth, res) = dfs(item)
//	            nlist_sum += res
//	            max_lv = max(max_lv, depth + 1)
//	        else:
//	            s += item
//	    nlist_sum += s * max_lv
//	    return (max_lv, nlist_sum)
	
	public static int revSum(List<Object> list){
		int[] dep={0};
		int[] res={0};
		dfsHelper(list, dep, res);
		return res[0];
	}
	
	public static void dfsHelper(List<Object> list, int[] dep, int res[]){
		for(int i=0;i<list.size();i++){
			if(list.get(i) instanceof List){
				dep[0] +=1;
				dfsHelper((List<Object>)list.get(i), dep, res);
			}
		}
		
	}

	public static List<List<Integer>> levelOrderTraversal(TreeNode root) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		if (root == null)
			return res;
		Queue<TreeNode> que = new LinkedList<TreeNode>();
		int curlevel = 0;
		int nextlevel = 0;
		que.offer(root);
		curlevel++;
		List<Integer> level = new ArrayList<Integer>();
		while (!que.isEmpty()) {
			TreeNode top = que.poll();
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
				res.add(level);
				level = new ArrayList<Integer>();
				curlevel = nextlevel;
				nextlevel = 0;
			}
		}
		return res;
	}

	public static List<List<Integer>> printFactors(int num) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> factors = new ArrayList<Integer>();
		for (int i = 2; i <= num / 2; i++)
			factors.add(i);
		List<Integer> sol = new ArrayList<Integer>();
		printFactorsUtil(0, factors, num, sol, res);
		res.add(0, new ArrayList<Integer>(Arrays.asList(1, num)));
		return res;
	}

	public static void printFactorsUtil(int dep, List<Integer> factors,
			int num, List<Integer> sol, List<List<Integer>> res) {
		if (num == 1) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}
		for (int i = dep; i < factors.size(); i++) {
			int factor = factors.get(i);
			if (factor <= num && num % factor == 0) {
				sol.add(factor);
				printFactorsUtil(i, factors, num / factor, sol, res);
				sol.remove(sol.size() - 1);
			}
		}
	}

	public static void mirror(TreeNode root) {
		if (root == null)
			return;
		TreeNode left = root.left;
		root.left = root.right;
		root.right = left;
		mirror(root.left);
		mirror(root.right);
	}

	// do post order traversal
	public static void mirror2(TreeNode root) {
		if (root == null)
			return;
		mirror2(root.left);
		mirror2(root.right);
		TreeNode left = root.left;
		root.left = root.right;
		root.right = left;
	}

	public static int sqrt(int x) {
		if (x < 0)
			return -1;

		double beg = 0;
		double end = x / 2 + 1;
		while (beg <= end) {
			double mid = (beg + end) / 2;
			double sq = mid * mid;
			if (sq == x)
				return (int) mid;
			if (sq > x)
				end = mid - 1;
			else
				beg = mid + 1;
		}
		return (int) end;
	}

	public static int[] searchRange(int[] A, int target) {
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
		res[1] = end;
		return res;
	}

	public int searchRotated(int[] A, int target) {
		int beg = 0;
		int end = A.length - 1;
		while (beg <= end) {
			int mid = (beg + end) / 2;
			if (A[mid] == target)
				return mid;
			else {
				if (A[mid] >= A[beg]) {
					if (A[beg] <= target && target < A[mid])
						end = mid - 1;
					else
						beg = mid + 1;
				} else if (A[mid] <= A[end]) {
					if (A[mid] < target && target <= A[end])
						beg = mid + 1;
					else
						end = mid - 1;
				}
			}
		}
		return -1;
	}

	public int maxSubArray(int[] A) {
		int sum = 0;
		int max = Integer.MIN_VALUE;
		for (int i = 0; i < A.length; i++) {
			sum += A[i];
			if (sum < 0)
				sum = 0;
			if (sum > max)
				max = sum;
		}
		if (max == 0) {
			max = A[0];
			for (int i = 1; i < A.length; i++)
				max = Math.max(max, A[i]);
		}
		return max;
	}

	public int maxProduct(int[] A) {
		int cur_max = A[0];
		int cur_min = A[0];
		int maxProd = A[0];

		for (int i = 1; i < A.length; i++) {
			int a = cur_max * A[i];
			int b = cur_min * A[i];
			cur_max = Math.max(Math.max(a, b), A[i]);
			cur_min = Math.min(Math.min(a, b), A[i]);
			maxProd = Math.max(maxProd, cur_max);
		}
		return maxProd;
	}

	public int maxProduct2(int[] A) {
		int curMax = 1;
		int curMin = 1;
		int max = Integer.MIN_VALUE;

		for (int i = 1; i < A.length; i++) {
			if (A[i] >= 0) {
				curMax = curMax <= 0 ? A[i] : curMax * A[i];
				curMin *= A[i];
			} else {
				int t = curMax;
				curMax = Math.max(A[i], curMin * A[i]);
				curMin = Math.min(t * A[i], A[i]);
			}
			max = Math.max(max, curMax);
		}
		return max;
	}

	public static List<Integer> iterator(Map<Integer, Object> map) {
		List<Integer> res = new ArrayList<Integer>();
		for (Integer i : map.keySet()) {
			Object obj = map.get(i);
			if (obj instanceof Integer)
				res.add((int) obj);
			else
				res.addAll(iterator((Map<Integer, Object>) obj));
		}
		return res;
	}

	public static int findCelebrity(int[][] matrix) {
		int n = matrix.length;
		Stack<Integer> stk = new Stack<Integer>();
		for (int i = 0; i < n; i++)
			stk.push(i);

		while (stk.size() != 1) {
			int A = stk.pop();
			int B = stk.pop();
			if (matrix[A][B] == 1)
				stk.push(B);
			else
				stk.push(A);
		}
		int celebrity = stk.pop();

		for (int i = 0; i < n; i++) {
			if (i != celebrity && matrix[celebrity][i] == 0)
				return -1;
		}
		return celebrity;
	}

	public static double pow(double x, int n) {
		if (n == 0)
			return 1;
		boolean neg = false;
		if (n < 0) {
			neg = true;
			n = -n;
		}
		double res = pow(x, n / 2);
		if (x % 2 == 0)
			res *= res;
		else
			res *= res * x;
		return neg ? 1 / res : res;
	}

	public int[] twoSum(int[] numbers, int target) {
		int[] res = { -1, -1 };
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		for (int i = 0; i < numbers.length; i++) {
			map.put(numbers[i], i + 1);
		}

		for (int i = 0; i < numbers.length; i++) {
			if (map.containsKey(target - numbers[i])) {
				res[0] = i + 1;
				res[1] = map.get(target - numbers[i]);
				if (res[1] > res[0])
					break;
			}
		}
		return res;
	}

	public List<List<Integer>> combinationSum(int[] candidates, int target) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		Arrays.sort(candidates);
		combinationSumUtil(0, candidates, target, sol, res, 0);
		return res;
	}

	public void combinationSumUtil(int dep, int[] candidates, int target,
			List<Integer> sol, List<List<Integer>> res, int cursum) {
		if (dep == candidates.length || cursum > target)
			return;
		if (cursum == target) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}
		for (int i = dep; i < candidates.length; i++) {
			cursum += candidates[i];
			sol.add(candidates[i]);
			combinationSumUtil(i, candidates, target, sol, res, cursum);
			cursum -= candidates[i];
			sol.remove(sol.size() - 1);
		}
	}

	public List<List<Integer>> combinationSum2(int[] num, int target) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		boolean[] used = new boolean[num.length];
		Arrays.sort(num);
		combinationSum2Util(0, num, target, used, sol, res, 0);
		return res;
	}

	public void combinationSum2Util(int cur, int[] num, int target,
			boolean[] used, List<Integer> sol, List<List<Integer>> res,
			int cursum) {
		if (cur == num.length || cursum > target)
			return;
		if (cursum == target) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
			return;
		}

		for (int i = cur; i < num.length; i++) {
			if (!used[i]) {
				if (i != 0 && num[i] == num[i - 1] && !used[i - 1])
					continue;
				used[i] = true;
				cursum += num[i];
				sol.add(num[i]);
				combinationSum2Util(i, num, target, used, sol, res, cursum);
				cursum -= num[i];
				used[i] = false;
				sol.remove(sol.size() - 1);
			}
		}
	}

	public static TreeNode LCAncestorBST(TreeNode root, TreeNode node1,
			TreeNode node2) {
		if (root == null || node1 == null || node2 == null)
			return null;
		if (Math.max(node1.val, node2.val) < root.val)
			return LCAncestorBST(root.left, node1, node2);
		else if (Math.min(node1.val, node2.val) > root.val)
			return LCAncestorBST(root.right, node1, node2);
		return root;
	}

	public static TreeNode LCAncestorBSTIterative(TreeNode root,
			TreeNode node1, TreeNode node2) {
		if (root == null || node1 == null || node2 == null)
			return null;
		TreeNode cur = root;
		while (cur != null) {
			if (Math.max(node1.val, node2.val) < cur.val)
				cur = cur.left;
			else if (Math.min(node1.val, node2.val) > cur.val)
				cur = cur.right;
			else
				break;
		}
		return cur;
	}

	public static TreeNode LCAncestorBT(TreeNode root, TreeNode node1,
			TreeNode node2) {
		if (root == null)
			return null;
		if (node1 == root || node2 == root)
			return root;
		TreeNode left_lac = LCAncestorBT(root.left, node1, node2);
		TreeNode right_lca = LCAncestorBT(root.right, node1, node2);
		if (left_lac != null && right_lca != null)
			return root;
		return left_lac != null ? left_lac : right_lca;
	}

	public static ListNode findIntersection(ListNode head1, ListNode head2) {
		if (head1 == null || head2 == null)
			return null;
		ListNode cur1 = head1;
		int len1 = 0;
		int len2 = 0;
		while (cur1 != null) {
			len1++;
			cur1 = cur1.next;
		}
		ListNode cur2 = head2;
		while (cur2 != null) {
			len2++;
			cur2 = cur2.next;
		}
		cur1 = len1 > len2 ? head1 : head2;
		cur2 = len1 > len2 ? head2 : head1;
		for (int i = 0; i < Math.abs(len1 - len2); i++)
			cur1 = cur1.next;
		while (cur1 != cur2) {
			cur1 = cur1.next;
			cur2 = cur2.next;
		}
		return cur1;

	}

	public static TreeNodeP LCAncestorWithParent(TreeNodeP node1,
			TreeNodeP node2) {
		if (node1 == null || node2 == null)
			return null;
		HashSet<TreeNodeP> set = new HashSet<TreeNodeP>();
		while (node1 != null || node2 != null) {
			if (node1 != null) {
				if (set.contains(node1))
					return node1;
				node1 = node1.parent;
			}
			if (node2 != null) {
				if (set.contains(node2))
					return node2;
				node2 = node2.parent;
			}
		}
		return null;
	}

	public static int getHeight(TreeNodeP node) {
		if (node == null)
			return 0;
		int h = 0;
		while (node != null) {
			h++;
			node = node.parent;
		}
		return h;
	}

	public static TreeNodeP LCAncestorWithParent2(TreeNodeP node1,
			TreeNodeP node2) {
		if (node1 == null || node2 == null)
			return null;
		int h1 = getHeight(node1);
		int h2 = getHeight(node2);

		if (h2 > h1) {
			TreeNodeP t = node1;
			node1 = node2;
			node2 = t;
		}

		for (int i = 0; i < Math.abs(h1 - h2); i++)
			node1 = node1.parent;

		while (node1 != null && node2 != null) {
			if (node1 == node2)
				return node1;
			node1 = node1.parent;
			node2 = node2.parent;
		}
		return null;

	}

	public static TreeNode buildBST(int[] preorder) {
		return buildBST(preorder, 0, preorder.length - 1);
	}

	public static TreeNode buildBST(int[] preorder, int beg, int end) {
		if (beg > end)
			return null;
		TreeNode root = new TreeNode(preorder[beg]);
		if (beg == end)
			return root;
		int index = end + 1;
		for (int i = beg + 1; i <= end; i++) {
			if (preorder[i] > root.val) {
				index = i;
				break;
			}
		}
		root.left = buildBST(preorder, beg + 1, index - 1);
		root.right = buildBST(preorder, index, end);
		return root;
	}

	public TreeNode buildTree(int[] preorder, int[] inorder) {
		int n = preorder.length;
		return buildTreeUtil(preorder, 0, n - 1, inorder, 0, n - 1);
	}

	public TreeNode buildTreeUtil(int[] preorder, int beg1, int end1,
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
		int len = index - beg2;
		root.left = buildTreeUtil(preorder, beg1 + 1, beg1 + len, inorder,
				beg2, index - 1);
		root.right = buildTreeUtil(preorder, beg1 + 1 + len, end1, inorder,
				index + 1, end2);
		return root;
	}

	public static String intToRoman(int num) {
		if (num <= 0 || num > 3999)
			return "";
		String[] roman = { "M", "CM", "D", "CD", "C", "XC", "L", "XL", "X",
				"IX", "V", "IV", "I" };
		int[] nums = { 1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1 };
		String res = "";
		for (int i = 0; i < nums.length; i++) {
			while (num >= nums[i]) {
				res += roman[i];
				num -= nums[i];
			}
		}
		return res;
	}

	public int romanToInt(String s) {
		Map<Character, Integer> map = new HashMap<Character, Integer>();
		map.put('I', 1);
		map.put('V', 5);
		map.put('X', 10);
		map.put('L', 50);
		map.put('C', 100);
		map.put('D', 500);
		map.put('M', 1000);
		int res = 0;
		for (int i = 0; i < s.length(); i++) {
			res += sign(map, s, i) * map.get(s.charAt(i));
		}
		return res;
	}

	public int sign(Map<Character, Integer> map, String s, int i) {
		if (i == s.length() - 1)
			return 1;
		if (map.get(s.charAt(i)) < map.get(s.charAt(i + 1)))
			return -1;
		return 1;
	}

	public int evalRPN(String[] tokens) {
		Stack<Integer> stk = new Stack<Integer>();
		for (int i = 0; i < tokens.length; i++) {
			String token = tokens[i];
			if (!token.equals("+") && !token.equals("-") && !token.equals("*")
					&& !token.equals("/"))
				stk.push(Integer.parseInt(token));
			else {
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

	// two arrays are sorted
	public static List<Integer> findIntersections(int[] A, int[] B) {
		List<Integer> res = new ArrayList<Integer>();
		int i = 0;
		int j = 0;
		while (i < A.length && j < B.length) {
			if (A[i] == B[j]) {
				res.add(A[i]);
				i++;
				j++;
			} else if (A[i] < B[j])
				i++;
			else
				j++;
		}
		return res;
	}

	public static List<Integer> findUnion(int[] A, int[] B) {
		List<Integer> res = new ArrayList<Integer>();
		int i = 0;
		int j = 0;
		while (i < A.length && j < B.length) {
			if (A[i] < B[j])
				res.add(A[i++]);
			else if (A[i] > B[j])
				res.add(B[j++]);
			else {
				res.add(A[i++]);
				j++;
			}
		}
		while (i < A.length)
			res.add(A[i++]);
		while (j < B.length)
			res.add(B[j++]);
		return res;
	}

	// two arrays are unsorted

	public static List<Integer> findIntersectionsUnsorted(int[] A, int[] B) {
		Set<Integer> set = new HashSet<Integer>();
		for (int i = 0; i < A.length; i++)
			set.add(A[i]);

		List<Integer> res = new ArrayList<Integer>();
		for (int i = 0; i < B.length; i++)
			if (set.contains(B[i]))
				res.add(B[i]);
		return res;
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
				ComplexNode t = cur.child;
				while (t.next != null)
					t = t.next;
				tail = t;
			}
			cur = cur.next;
		}
		return head;
	}

	// if four ways, add set to check if visited already
	public static ComplexNode flattenListAllDirections(ComplexNode head) {
		if (head == null)
			return null;
		ComplexNode tail = head;
		while (tail.next != null)
			tail = tail.next;
		ComplexNode cur = head;
		while (cur != tail) {
			if (cur.child != null) {
				tail.next = cur.child;
				ComplexNode t1 = cur.child;
				while (t1.next != null)
					t1 = t1.next;
				tail = t1;
			}
			if (cur.parent != null) {
				tail.next = cur.parent;
				ComplexNode t2 = cur.parent;
				while (t2.next != null)
					t2 = t2.next;
				tail = t2;
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
		List<Interval> res = new ArrayList<Interval>();
		Collections.sort(intervals, new IntervalComparator());
		res.add(intervals.get(0));
		for (int i = 1; i < intervals.size(); i++) {
			Interval last = res.get(res.size() - 1);
			Interval cur_interval = intervals.get(i);
			if (cur_interval.start > last.end)
				res.add(cur_interval);
			else {
				last.end = Math.max(last.end, cur_interval.end);
			}
		}
		return res;
	}

	public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
		if (intervals.size() == 0) {
			intervals.add(newInterval);
			return intervals;
		}
		boolean inserted = false;
		List<Interval> res = new ArrayList<Interval>();
		for (int i = 0; i < intervals.size(); i++) {
			Interval interval = intervals.get(i);
			if (interval.start < newInterval.start)
				insertInterval(interval, res);
			else {
				insertInterval(newInterval, res);
				insertInterval(interval, res);
				inserted = true;
			}
		}
		if (!inserted)
			insertInterval(newInterval, res);
		return res;
	}

	public void insertInterval(Interval interval, List<Interval> res) {
		if (res.size() == 0)
			res.add(interval);
		else {
			Interval last = res.get(res.size() - 1);
			if (last.end < interval.start)
				res.add(interval);
			else
				last.end = Math.max(last.end, interval.end);
		}
	}

	public List<List<Integer>> permute(int[] num) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		boolean[] used = new boolean[num.length];
		permuteUtil(0, num, used, sol, res);
		return res;
	}

	public void permuteUtil(int dep, int[] num, boolean[] used,
			List<Integer> sol, List<List<Integer>> res) {
		if (dep == num.length) {
			List<Integer> out = new ArrayList<Integer>(sol);
			res.add(out);
		}

		for (int i = 0; i < num.length; i++) {
			if (!used[i]) {
				used[i] = true;
				sol.add(num[i]);
				permuteUtil(dep + 1, num, used, sol, res);
				used[i] = false;
				sol.remove(sol.size() - 1);
			}
		}
	}

	public List<List<Integer>> permuteUnique(int[] num) {
		List<List<Integer>> res = new ArrayList<List<Integer>>();
		List<Integer> sol = new ArrayList<Integer>();
		boolean[] used = new boolean[num.length];
		Arrays.sort(num);
		permuteUniqueUtil(0, num, used, sol, res);
		return res;
	}

	public void permuteUniqueUtil(int dep, int[] num, boolean[] used,
			List<Integer> sol, List<List<Integer>> res) {
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
				permuteUniqueUtil(dep + 1, num, used, sol, res);
				used[i] = false;
				sol.remove(sol.size() - 1);
			}
		}
	}

	// perfect shuffle
	public static void shuffle(int[] A) {
		if (A.length < 2)
			return;
		Random r = new Random();
		for (int i = A.length - 1; i > 0; i--) {
			int j = r.nextInt(i + 1);
			int t = A[i];
			A[i] = A[j];
			A[j] = t;
		}
		System.out.println(Arrays.toString(A));
	}

	public static boolean isIsomorphic(String s1, String s2) {
		if (s1.length() != s2.length())
			return false;
		Map<Character, Character> map1 = new HashMap<Character, Character>();
		Map<Character, Character> map2 = new HashMap<Character, Character>();
		for (int i = 0; i < s1.length(); i++) {
			char c1 = s1.charAt(i);
			char c2 = s2.charAt(i);
			if (map1.containsKey(c1) && map1.get(c1) != c2)
				return false;
			else
				map1.put(c1, c2);
			if (map2.containsKey(c2) && map2.get(c2) != c1)
				return false;
			else
				map2.put(c2, c1);
		}
		return true;
	}

	public static boolean isIsomorphic2(String s1, String s2) {
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

	// given array, out array each element if the product of all the elems
	// except itself
	public static int[] productWithoutItself(int[] A) {
		int n = A.length;
		int[] left = new int[n];
		int[] right = new int[n];
		int[] prod = new int[n];
		left[0] = 1;
		for (int i = 1; i < n; i++)
			left[i] = left[i - 1] * A[i - 1];
		right[n - 1] = 1;
		for (int i = n - 2; i >= 0; i--)
			right[i] = right[i + 1] * A[i + 1];

		for (int i = 0; i < n; i++)
			prod[i] = left[i] * right[i];
		System.out.println(Arrays.toString(prod));
		return prod;
	}

	// Find and print repeated sequences of 10 characters
	public static Set<String> repeatedSubstrings(String input, int len) {
		if (input.isEmpty() || len <= 0 || len >= input.length()) {
			throw new IllegalArgumentException();
		}
		Set<String> nonRepeatingSeq = new TreeSet<String>();
		Set<String> repeatingSeq = new TreeSet<String>();

		for (int i = 0; i < input.length() - len + 1; i++) {
			String s = input.substring(i, i + len);
			if (!nonRepeatingSeq.add(s))
				repeatingSeq.add(s);
		}
		System.out.println(nonRepeatingSeq);
		System.out.println(repeatingSeq);
		return repeatingSeq;
	}

	// Returns true if str1 is a subsequence of str2.
	public static boolean isSubSequence(String s1, String s2) {
		if (s1.length() > s2.length())
			return false;
		int i = 0;
		for (int j = 0; i < s1.length() && j < s2.length(); j++) {
			if (s1.charAt(i) == s2.charAt(j))
				i++;
		}
		return i == s1.length();
	}

	public static List<Object> flatten(List<?> list) {
		List<Object> res = new LinkedList<Object>();
		flatten(list, res);
		return res;
	}

	public static void flatten(List<?> list, List<Object> res) {
		for (Object item : list) {
			if (item instanceof List<?>)
				flatten((List<?>) item, res);
			else
				res.add(item);
		}
	}

	public static TreeNode buildTree(List<Relation> data) {
		if (data.size() == 0)
			return null;
		Map<Integer, TreeNode> map = new HashMap<Integer, TreeNode>();
		TreeNode root = null;
		for (int i = 0; i < data.size(); i++) {
			Relation relation = data.get(i);
			TreeNode node = new TreeNode(relation.child);
			map.put(relation.child, node);
		}

		for (int i = 0; i < data.size(); i++) {
			Relation relation = data.get(i);
			Integer child = relation.child;
			if (relation.parent == null) {
				root = map.get(relation.child);
				continue;
			}
			TreeNode parent = map.get(relation.parent);
			if (relation.isLeft)
				parent.left = map.get(child);
			else
				parent.right = map.get(child);
		}
		return root;
	}

	public static int shortest(String[] words, String word1, String word2) {
		int minDis = Integer.MAX_VALUE;
		int word1_pos = -1;
		int word2_pos = -1;
		for (int i = 0; i < words.length; i++) {
			String word = words[i];
			if (word.equals(word1)) {
				word1_pos = i;
				// Comment following lines if word order matters
				if (word2_pos != -1) {
					int dis = i - word2_pos;
					if (dis < minDis)
						minDis = dis;
				}
			} else if (word.equals(word2)) {
				word2_pos = i;
				if (word1_pos != -1) {
					int dis = word2_pos - word1_pos;
					if (dis < minDis)
						minDis = dis;
				}
			}
		}
		return minDis;
	}

	public static int numDecodings(String s) {
		if (s.length() == 0)
			return 0;
		int[] dp = new int[s.length() + 1];
		dp[0] = 1;
		if (s.charAt(0) > '0' && s.charAt(0) <= '9')
			dp[1] = 1;
		for (int i = 2; i <= s.length(); i++) {
			char c1 = s.charAt(i - 1);
			char c2 = s.charAt(i - 2);
			if (c1 > '0' && c1 <= '9') {
				dp[i] = dp[i - 1];
				System.out.println("dp at i= " + i + " is " + dp[i]);
			}
			System.out.println(c1 + " " + c2);
			if (c2 == '1' || (c2 == '2' && c1 <= '6')) {
				dp[i] = dp[i] + dp[i - 2];
				System.out.println("if two digits dp at i= " + i + " is "
						+ dp[i]);
			}
		}
		System.out.println(Arrays.toString(dp));
		return dp[s.length()];
	}

	public static List<String> restoreIpAddresses(String s) {
		List<String> res = new ArrayList<String>();
		if (s.length() < 4 || s.length() > 12)
			return res;
		restoreIp(0, s, "", res);
		return res;
	}

	public static void restoreIp(int dep, String s, String sol, List<String> res) {
		if (dep == 3 && isValidNum(s)) {
			res.add(sol + s);
			return;
		}

		for (int i = 1; i < 4 && i < s.length(); i++) {
			if (isValidNum(s.substring(0, i))) {
				restoreIp(dep + 1, s.substring(i), sol + s.substring(0, i)
						+ ".", res);
			}
		}
	}

	public static boolean isValidNum(String s) {
		if (s.charAt(0) == '0')
			return s.equals("0");
		int num = Integer.parseInt(s);
		return num >= 1 && num <= 255;
	}

	public int[][] generateMatrix(int n) {
		int[][] mat = new int[n][n];
		int num = 1;
		int top = 0;
		int bottom = n - 1;
		int left = 0;
		int right = n - 1;

		while (true) {
			for (int i = left; i <= right; i++)
				mat[top][i] = num++;
			if (++top > bottom)
				break;
			for (int i = top; i <= bottom; i++)
				mat[i][right] = num++;
			if (--right < left)
				break;
			for (int i = right; i >= left; i--)
				mat[bottom][i] = num++;
			if (--bottom < top)
				break;
			for (int i = bottom; i >= top; i--)
				mat[i][left] = num++;
			if (++left > right)
				break;
		}
		return mat;
	}

	public List<Integer> spiralOrder(int[][] matrix) {
		List<Integer> res=new ArrayList<Integer>();
		if(matrix.length==0)
			return res;
		int top=0;
		int bottom=matrix.length-1;
		int left=0;
		int right=matrix[0].length-1;
		
		while(true){
			for(int i=left;i<=right;i++)
				res.add(matrix[top][i]);
			if(++top>bottom)
				break;
			for(int i=top;i<=bottom;i++)
				res.add(matrix[i][right]);
			if(--right<left)
				break;
			for(int i=right;i>=left;i--)
				res.add(matrix[bottom][i]);
			if(--bottom<top)
				break;
			for(int i=bottom;i>=top;i--)
				res.add(matrix[i][left]);
			if(++left>right)
				break;
		}
		return res;
	}
	
	 public static String countAndSay(int n) {
	        if(n==0)
	            return "";
	        String res="1";
	        for(int i=1;i<=n;i++){
	            String tmp="";
	            int count=1;
	            char c=res.charAt(0);
	            for(int j=1;j<res.length();j++){
	                if(res.charAt(j)==c)
	                    count++;
	                else{
	                    tmp+=""+count+c;
	                    c=res.charAt(j);
	                    count=1;
	                }
	            }
	            tmp+=""+count+c;
	            res=tmp;
	        }
	        return res;
	    }
	 
	 
	 public boolean canPlaceFlowers(List<Boolean> flowerbed, int numberToPlace) {
			this.hashCode();
		    if(flowerbed == null || flowerbed.isEmpty()){
		        throw new IllegalArgumentException("bed is empty");
		    }
		    
		    if(numberToPlace==0)
		        return true;

		    if(flowerbed.size()==1){
		        return !flowerbed.get(0) && numberToPlace<=1;
		    }
		    
		    int counter = 0;
		    
		    for(int i=0; i< flowerbed.size(); i++){
		    	if(!flowerbed.get(i)){
		    		if((i==0 && !flowerbed.get(i+1)) || (i==flowerbed.size()-1 && !flowerbed.get(i-1)) || (!flowerbed.get(i+1) && !flowerbed.get(i-1)) ){
		    			flowerbed.set(i, true);
		    			counter++;
		    			if(counter==numberToPlace)
		    				return true;
		    		}
		    	}
		    }    
		    
		    return false;
		}

	/**
	 * . Given a nested list of integers, returns the sum of all integers in the
	 * list weighted by their depth For example, given the list {{1,1},2,{1,1}}
	 * the function should return 10 (four 1's at depth 2, one 2 at depth 1)
	 * Given the list {1,{4,{6}}} the function should return 27 (one 1 at depth
	 * 1, one 4 at depth 2, and one 6 at depth 3)
	 */

	// {1,2,{1,2}}
	// input.getInteger() =1
	// input.getInteger() =2
	// input.getInteger() = null?
	//
	// ArrayList<NestedInteger> input
	// public int depthSum (List<NestedInteger> input){
	//
	//
	// }

	//
	// 1st phone
	// 1, write Singleton class
	// 2,
	// /**
	// * Three segments of lengths A, B, C form a triangle iff .
	// *
	// * A + B > C.
	// * B + C > A
	// * A + C > B
	// *
	// * e.g.
	// * 6, 4, 5 can form a triangle.
	// * 10, 2, 7 can't
	// *
	// * Given a list of segments lengths algorithm should find at least one
	// triplet of segments that form a triangle (if any).
	// *
	// * Method should return an array of either:
	// * - 3 elements: segments that form a triangle (i.e. satisfy the condition
	// above)
	// * - empty array if there are no such segments
	// */.
	//
	// 3,
	// /**.
	// * Given a matrix of following relationships between N LinkedIn users
	// (with ids from 0 to N-1):
	// * followingMatrix[j] == true iff user i is following user j
	// * thus followingMatrix[j] doesn't imply followingMatrix[j].. visit
	// 1point3acres.com for more.
	// * Let's also agree that followingMatrix == false.
	// *
	// * An influencer is a user who is:
	// * - followed by everyone else and
	// * - not following anyone herself/himself
	// *
	// * This method should return the influencer's id in a given matrix of
	// following relationships,
	// * or return -1 if there is no influencer in this group.
	// */
	// //input: boolean[][] followingMatrix
	//
	// Solution:
	// * If A knows B, then A can’t be celebrity. Discard A, and B may be
	// celebrity.
	// * If A doesn’t know B, then B can’t be celebrity. Discard B, and A may be
	// celebrity.
	// * Repeat above two steps till we left with only one person.
	// * Ensure the remained person is celebrity. (Why do we need this step?
	// Maybe no one is celebrity)

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		TreeNode root = new TreeNode(5);
		root.left = new TreeNode(2);
		root.left.right = new TreeNode(4);
		root.left.right.left = new TreeNode(3);
		root.left.left = new TreeNode(1);

		root.right = new TreeNode(8);
		root.right.left = new TreeNode(6);
		root.right.left.right = new TreeNode(7);
		root.right.right = new TreeNode(9);
		root.right.right.left = new TreeNode(11);
		root.right.right.left.right = new TreeNode(13);

		String serialization = serializeBinaryTree(root);
		System.out.println(serialization);
		TreeNode root1 = deserialize(serialization);
		inorder(root1);
		System.out.println();

		mirror(root);
		inorder(root);
		System.out.println();

		List<Object> list = new ArrayList<Object>();
		list.add(1);
		list.add(3);
		List<Object> l1 = new ArrayList<Object>();
		l1.add(2);
		l1.add(4);
		List<Object> l2 = new ArrayList<Object>();
		l2.add(5);
		l2.add(2);
		l1.add(l2);
		list.add(l1);

		System.out.println(levelSum2(list));
		
		System.out.println(list);
		System.out.println("the deepest is "+getLevel(list));

		System.out.println(reverseLevelSum(list));
		
		System.out.println(printFactors(100));
		System.out.println(sqrt(144));

		Map<Integer, Object> map = new HashMap<Integer, Object>();
		map.put(1, 2);
		map.put(2, 4);
		HashMap<Integer, Integer> item1 = new HashMap<Integer, Integer>();
		item1.put(4, 16);
		item1.put(5, 25);
		map.put(3, item1);

		System.out.println(iterator(map));

		int[] preorder = { 5, 4, 2, 3, 8, 7, 9 };
		TreeNode r = buildBST(preorder);
		inorder(r);
		System.out.println();

		System.out.println(intToRoman(101));

		int arr1[] = { 1, 3, 4, 5, 7 };
		int arr2[] = { 2, 3, 5, 6 };
		System.out.println(findIntersections(arr1, arr2));
		System.out.println(findIntersectionsUnsorted(arr1, arr2));
		System.out.println(findUnion(arr1, arr2));

		int[] majorityArr = { 3, 2, 3, 4, 3, 5, 3, 2, 3 };
		System.out.println(findMajority(majorityArr));

		int arr[] = { 1, 2, 3, 4, 5, 6, 7, 8 };
		shuffle(arr);

		System.out.println(isIsomorphic("foo", "app"));
		System.out.println(isIsomorphic("bar", "foo"));
		System.out.println(isIsomorphic("turtle", "tletur"));
		System.out.println(isIsomorphic("ab", "ca"));

		System.out.println(isIsomorphic2("foo", "app"));
		System.out.println(isIsomorphic2("bar", "foo"));
		System.out.println(isIsomorphic2("turtle", "tletur"));
		System.out.println(isIsomorphic2("ab", "ca"));

		int[] A = { 10, 2, 5 };
		productWithoutItself(A);

		repeatedSubstrings("ABCACBABC", 3);
		repeatedSubstrings("ABCABCA", 2);
		// repeatedSubstrings( "ABCACBABC" );

		System.out.println(isSubSequence("AXY", "ADXCPY"));

		ComplexNode head = new ComplexNode(1);
		head.child = new ComplexNode(4);
		head.parent = new ComplexNode(5);
		ComplexNode node1 = new ComplexNode(2);
		node1.parent = new ComplexNode(6);
		node1.parent.parent = new ComplexNode(7);
		node1.parent.parent.next = new ComplexNode(8);
		node1.parent.parent.next.next = new ComplexNode(9);
		node1.parent.parent.next.next.parent = new ComplexNode(11);
		node1.parent.parent.next.next.child = new ComplexNode(10);
		head.next = node1;
		ComplexNode node2 = new ComplexNode(3);
		node1.next = node2;
		node2.child = new ComplexNode(12);
		node2.child.next = new ComplexNode(13);
		node2.child.next.parent = new ComplexNode(15);
		node2.child.next.child = new ComplexNode(14);

		ComplexNode flattenedHead = flattenListAllDirections(head);
		while (flattenedHead != null) {
			System.out.print(flattenedHead.val + " ");
			flattenedHead = flattenedHead.next;
		}
		System.out.println();
		// Child Parent IsLeft
		// 15 20 true
		// 19 80 true
		// 17 20 false
		// 16 80 false
		// 80 50 false
		// 50 null false
		// 20 50 true

		Relation r1 = new Relation(15, 20, true);
		Relation r2 = new Relation(19, 80, true);
		Relation r3 = new Relation(17, 20, false);
		Relation r4 = new Relation(16, 80, false);
		Relation r5 = new Relation(80, 50, false);
		Relation r6 = new Relation(50, null, false);
		Relation r7 = new Relation(20, 50, true);
		List<Relation> l = new ArrayList<Relation>();
		l.add(r1);
		l.add(r2);
		l.add(r3);
		l.add(r4);
		l.add(r5);
		l.add(r6);
		l.add(r7);

		TreeNode rootWithP = buildTree(l);
		inorder(rootWithP);
		System.out.println(levelOrderTraversal(rootWithP));

		String nums[] = { "3", "5", "4", "2", "6", "3", "0", "0", "5", "4",
				"8", "3" };
		System.out.println(shortest(nums, "5", "3"));

		System.out.println(numDecodings("26"));

		System.out.println(restoreIpAddresses("1111"));
		
		System.out.println(countAndSay(1));
	}

}
