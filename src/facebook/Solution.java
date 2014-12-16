package facebook;

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
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Set;
import java.util.Stack;




public class Solution {
	public List<String> letterCombinations(String digits) {
		List<String> res=new ArrayList<String>();
		dfs(digits, 0, "", res);
		return res;
	}
	
	public void dfs(String digits, int cur, String sol, List<String> res){
		if(cur==digits.length()){
			res.add(sol);
			return;
		}
		String s=getString(digits.charAt(cur)-'0');
		for(int i=0;i<s.length();i++){
			dfs(digits,cur+1,sol+s.charAt(i),res);
		}
	}
	
	public String getString(int num){
		String[] strs={"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
		return strs[num];
	}
	
	
	 public boolean hasPathSum(TreeNode root, int sum) {
		 if(root==null)
			 return false;
		 return hasPathSum(root, sum, 0);
	 }
	 
	 public boolean hasPathSum(TreeNode root, int sum, int cursum){
		 if(root==null)
			 return false;
		 cursum+=root.val;
		 if(root.left==null&&root.right==null&&cursum==sum)
			 return true;
		 return hasPathSum(root.left,sum,cursum)||hasPathSum(root.right,sum,cursum);
	 }
	 
	 public ListNode reverseList(ListNode head){
		 if(head==null||head.next==null)
			 return head;
		 ListNode cur=head;
		 ListNode pre=null;
		 while(cur!=null){
			 ListNode pnext=cur.next;
			 cur.next=pre;
			 pre=cur;
			 cur=pnext;
		 }
		 return pre;
	 }
	 
	 public ListNode reverseListRecur(ListNode head){
		 if(head==null||head.next==null)
			 return head;
		 ListNode pnext=head.next;
		 head.next=null;
		 ListNode node=reverseListRecur(pnext);
		 pnext.next=head;
		 return node;		 
	 }
	 
	// decode
		public int decodeWays(String s){
			int[] ways={0};
			decodeWaysUtil(s, ways);
			return ways[0];
		}
		
		public void decodeWaysUtil(String s, int[]ways){
			if(s.length()==0)
				ways[0]++;
			for(int i=0;i<=1&&i<s.length();i++){
				if(isValidNum(s.substring(0, i+1)))
					decodeWaysUtil(s.substring(i+1),ways);
			}
		}
		
		public boolean isValidNum(String s){
			if(s.charAt(0)=='0')
				return false;
			int num=Integer.parseInt(s);
			return num>0&&num<=26;
		}
		
		public int numDecodingsDP(String s) {
			if(s.length()==0||s.charAt(0)=='0')
				return 0;
			if(s.length()==1)
				return s.charAt(0)=='0'?0:1;
			int total=1;
			int last1=1;
			int last2=1;
			for(int i=2;i<=s.length();i++){
				if(isValidNum(s.substring(i-1,i)))
					total=last1;
				else
					total=0;
				if(isValidNum(s.substring(i-2,i)))
						total+=last2;
				last2=last1;
				last1=total;
			}
			return total;
		}
		
		public int numDecodingsDP2(String s) {
			if(s.length()==0||s.charAt(0)=='0')
				return 0;
			int total=1;
			int last1=1;
			int last2=1;
			for(int i=1;i<s.length();i++){
				total=s.charAt(i)=='0'?0:last1;
				if(s.charAt(i-1)=='1'||s.charAt(i-1)=='2'&&s.charAt(i)<'7')
					total+=last2;
				last2=last1;
				last1=total;
			}
			return total;
		}
	 
	 // given two arrays, rearrange first array according to the second array
	public void reArrange(char[] A, int[] index) {
		// char[] temp=new char[A.length];
		// for(int i=0;i<A.length;i++){
		// temp[index[i]]=A[i];
		// }
		for (int i = 0; i < A.length; i++) {
			while (index[i] != i) {
				int idx = index[i];
				char c = A[idx];
				A[idx] = A[i];
				A[i] = c;

				int t = index[idx];
				index[idx] = idx;
				index[i] = t;
			}

		}
		System.out.println(Arrays.toString(A));
	}
	
	public void printTreeVertical(TreeNode root){
		if(root==null)
			return;
		HashMap<Integer, List<Integer>> map=new HashMap<Integer, List<Integer>>();
		printTreeHelper(root, map, 0);
		
		Iterator<Integer> it=map.keySet().iterator();
		while(it.hasNext()){
			int idex=it.next();
			System.out.println(map.get(idex));
		}
	}
	
	public void printTreeHelper(TreeNode root, HashMap<Integer, List<Integer>> map, int index){
		if(root==null)
			return;
		if(map.containsKey(index))
			map.get(index).add(root.val);
		else{
			List<Integer> lst=new ArrayList<Integer>();
			lst.add(root.val);
			map.put(index, lst);
		}
		printTreeHelper(root.left,map,index-1);
		printTreeHelper(root.right,map,index+1);
		
	}
	
	public int sqrt(int x){
		if(x==1)
			return 1;
		double last=0;
		double res=x;
		while(last!=res){
			last=res;
			res=(res+x/res)/2;
		}
		return (int) res;
	}
//	lca
	
	public TreeNode lowestCommonAncestor(TreeNode root, TreeNode node1,TreeNode node2){
		if(root==null||node1==null||node1==null)
			return null;
		if(root==node1||root==node2)
			return root;
		TreeNode leftLca=lowestCommonAncestor(root.left,node1,node2);
		TreeNode rightLca=lowestCommonAncestor(root.right,node1,node2);
		if(leftLca!=null&&rightLca!=null)
			return root;
		return leftLca!=null?leftLca:rightLca;
	}
	
	// merge k lists
	public ListNode mergeKLists(List<ListNode> lists) {
		if(lists.size()==0)
			return null;
		Comparator<ListNode> comp=new Comparator<ListNode>(){

			@Override
			public int compare(ListNode o1, ListNode o2) {
				// TODO Auto-generated method stub
				return o1.val-o2.val;
			}
			
		};
		PriorityQueue<ListNode> heap=new PriorityQueue<ListNode>(lists.size(),comp);
		ListNode dummy=new ListNode(0);
		ListNode pre=dummy;
		for(ListNode node:lists){
			if(node!=null)
				heap.offer(node);
		}
		while(!heap.isEmpty()){
			ListNode node=heap.poll();
			pre.next=node;
			pre=pre.next;
			if(node.next!=null)
				heap.offer(node.next);
		}
		return dummy.next;
	}
	
	// if one array is a subset of the other
	//mothod 1: two simple loops --->n2
	// method2: sort first array, for each element in second array, do binary search----nlogn+mlogn
	// method3: hashing hash first array, do linear search of second array
	// method 4: sort both arrays, two pointers
	// implement method 3 and 4
	
	public boolean isSubset3(int[] A, int[] B){
		HashSet<Integer> set=new HashSet<Integer>();
		for(int i:A)
			set.add(i);
		for(int i=0;i<B.length;i++){
			if(!set.contains(B[i]))
				return false;
		}
		return true;
	}
	
	public boolean isSubset4(int[] A, int[] B){
		Arrays.sort(A);
		Arrays.sort(B);
		int i=0;
		int j=0;
		while(i<A.length&&j<B.length){
			if(A[i]<B[j])
				i++;
			else if(A[i]==B[j]){
				i++;
				j++;
			}
			else
				return false;
		}
		return true;
	}
	
	// bst
	//level order
	public List<List<Integer>> levelOrder(TreeNode root) {
		List<List<Integer>> res=new ArrayList<List<Integer>>();
		if(root==null)
			return res;
		Queue<TreeNode> que=new LinkedList<TreeNode>();
		int curlevel=0;
		int nextlevel=0;
		que.add(root);
		curlevel++;
		List<Integer> level=new ArrayList<Integer>();
		while(!que.isEmpty()){
			TreeNode node=que.remove();
			curlevel--;
			level.add(node.val);
			if(node.left!=null){
				que.offer(node.left);
				nextlevel++;
			}
			if(node.right!=null){
				que.offer(node.right);
				nextlevel++;
			}
			if(curlevel==0){
				res.add(level);
				level=new ArrayList<Integer>();
				curlevel=nextlevel;
				nextlevel=0;
			}
		}
		return res;
	}
	// division
	public int divide(int dividend, int divisor) {
		boolean neg=false;
		if(dividend>0&&divisor<0||dividend<0&&divisor>0)
			neg=true;
		long a=Math.abs((long)dividend);
		long b=Math.abs((long)divisor);
		int res=0;
		while(a>=b){
			int shift=0;
			while((b<<shift)<=a)
				shift++;
			res+=1<<(shift-1);
			a-=b<<(shift-1);
		}
		return neg?-res:res;
	}
	
	// Add two binary numbers (Input as a string)
	
	public String addBinary(String a, String b){
		if(a.isEmpty()||b.isEmpty())
			return a.isEmpty()?b:a;
		String res="";
		int i=a.length()-1;
		int j=b.length()-1;
		int carry=0;
		while(i>=0&&j>=0){
			int sum=a.charAt(i)-'0'+b.charAt(j)-'0'+carry;
			carry=sum/2;
			sum=sum%2;
			res=sum+res;
			i--;
			j--;
			System.out.println("carry is "+ carry+", sum is "+sum+", res is "+res);
		}
		while(i>=0){
			int sum=a.charAt(i)-'0'+carry;
			carry=sum/2;
			sum=sum%2;
			res=sum+res;
			i--;
		}
		while(j>=0){
			int sum=b.charAt(j)-'0'+carry;
			carry=sum/2;
			sum=sum%2;
			res=sum+res;	
			j--;
		}
		if(carry==1)
			res="1"+res;
		return res;
	}
	
	// facebook remove dups
    public static String removeDuplicates(String s){
    	if(s.length()<2)
    		return s;
    	String res="";
    	int[] map=new int[256];
    	for(int i=0;i<s.length();i++){
    		if(map[s.charAt(i)]==0){
    			map[s.charAt(i)]=1;
    			res+=s.charAt(i);
    		}
    	}
    	return res;
    }
    
    public int removeDuplicates(int[] num){
    	HashSet<Integer> set=new HashSet<Integer>();
    	int j=0;
    	for(int i=0;i<num.length;i++){
    		if(!set.contains(num[i])){
    			set.add(num[i]);
    			num[j++]=num[i];
    		}
    	}
//    	for(;j<num.length;j++)
//    		num[j]=0;
    	return j;
    }
    
 // oneEditAway
    public static boolean oneEditAway(String s1, String s2){
    	String small=s1.length()<s2.length()?s1:s2;
    	String large=s1.length()<s2.length()?s2:s1;
    	
    	int edit=0;
    	if(large.length()-small.length()>1)
    		return false;
    	else if(large.length()-small.length()==1){
    		int i=0;
    		while(i<small.length()){
    			if(small.charAt(i)!=large.charAt(i+edit)){
    				if(++edit>1)
    					return false;
    			}
    			else
    				i++;
    		}
    	}
    	else{
    		for(int i=0;i<small.length();i++){
    			if(small.charAt(i)!=large.charAt(i)){
    				if(++edit>1)
    					return false;
    			}
    		}
    	}
    	return true;
    }
    
    
//	Sink Zero in Binary Tree. Swap zero value of a node with non-zero value of one of its descendants 
//	so that no node with value zero could be parent of node with non-zero.
    
    public void sinkZeroNode(TreeNode root){
    	if(root==null)
    		return;
    	if(root.val==0){
    		if(root.left!=null&&root.left.val!=0){
    			root.val=root.left.val;
    			root.left.val=0;
    			System.out.println("root val(l) now is"+root.val);
    		}
    		else if(root.right!=null&&root.right.val!=0){
    			root.val=root.right.val;
    			root.right.val=0;
    		}
    	}
    	sinkZeroNode(root.left);
    	sinkZeroNode(root.right);
    }
    
    public void inorder(TreeNode root){
    	if(root==null)
    		return;
    	inorder(root.left);
    	System.out.print(root.val+" ");
    	inorder(root.right);
    }
    
	
//	kth smallest number in a binary search tree
    public int kthSmallestNode(TreeNode root, int k){
    	if(root==null)
    		return Integer.MAX_VALUE;
    	int count_left=countNodes(root.left);
    	if(count_left==k-1)
    		return root.val;
    	else if(count_left>k-1)
    		return kthSmallestNode(root.left,k);
    	else
    		return kthSmallestNode(root.right,k-count_left-1);
    }
    
    public int countNodes(TreeNode root){
    	if(root==null)
    		return 0;
    	return countNodes(root.left)+countNodes(root.right)+1;
    }
    
//	compare two general trees and making sure they have the same elements.
    public boolean isSameTree(TreeNode p, TreeNode q){
    	if(p==null&&q==null)
    		return true;
    	if(p==null||q==null)
    		return false;
    	if(p.val!=q.val)
    		return false;
    	return isSameTree(p.left,q.left)&&isSameTree(p.right,q.right);
    }
    
//	rotated binary search    
    public int search(int[] A, int target) {
    	int beg=0;
    	int end=A.length-1;
    	while(beg<=end){
    		int mid=(beg+end)/2;
    		if(A[mid]==target)
    			return mid;
    		else if(A[mid]>=A[beg]){
    			if(A[beg]<=target&&target<A[mid])
    				end=mid-1;
    			else
    				beg=mid+1;
    		}
    		else{
    			if(A[mid]<target&&target<=A[end])
    				beg=mid+1;
    			else
    				end=mid-1;
    		}
    	}
    	return -1;
    }
    
//	1. Function to check if 2 words are anagrams
    public boolean anagram(String s1, String s2){
    	if(s1.length()!=s2.length())
    		return false;
    	char[] chars1=s1.toCharArray();
    	Arrays.sort(chars1);
    	char[] chars2=s2.toCharArray();
    	Arrays.sort(chars2);
    	String s11=new String(chars1);
    	String s21=new String(chars2);
    	if(s11.equals(s21))
    		return true;
    	return false;
    }
    
    public boolean anagram2(String s1, String s2){
    	if(s1.length()!=s2.length())
    		return false;
    	int[] map1=new int[256];
    	int[] map2=new int[256];
    	for(int i=0;i<s1.length();i++){
    		map1[s1.charAt(i)]++;
    		map2[s2.charAt(i)]++;
    	}
    	for(int i=0;i<256;i++){
    		if(map1[i]!=map2[i])
    			return false;
    	}
    	return true;
    }
    
//	2. Function to check if any 3 numbers sum to x.
    public List<List<Integer>> threeSum(int[] num) {
    	List<List<Integer>> res=new ArrayList<List<Integer>>();
    	if(num.length<3)
    		return res;
    	Arrays.sort(num);
    	for(int i=0;i<num.length-2;i++){
    		if(i==0||num[i]>num[i-1]){
    			int j=i+1;
        		int k=num.length-1;
        		while(j<k){
        			int sum=num[i]+num[j]+num[k];
        			if(sum==0){
        				List<Integer> sol=new ArrayList<Integer>();
        				sol.add(num[i]);
        				sol.add(num[j]);
        				sol.add(num[k]);
        				res.add(sol);
        				j++;
        				k--;
        				while(k>j&&num[k]==num[k+1])
        					k--;
        				while(j<k&&num[j]==num[j-1])
        					j++;
        			}
        			else if(sum>0){
        				k--;
        			}
        			else{
        				j++;
        			}
        		}
    		}
    		
    	}
    	return res;
    }
    
//	Return the length of the longest sequence of increasing numbers in an unsorted array
	public int longestIncreasingSeq(int[] A){
		int[] dp=new int[A.length];
		for(int i=0;i<A.length;i++)
			dp[i]=1;
		for(int i=1;i<A.length;i++){
			for(int j=0;j<i;j++){
				if(A[i]>A[j]&&dp[i]<dp[j]+1)
					dp[i]=dp[j]+1;
			}
		}
		int max=1;
		for(int i=0;i<dp.length;i++)
			max=Math.max(max, dp[i]);
		return max;
	}
	
//	given a m * n grids, and one is allowed to move up or right, find the number of paths between two grids.
	public int uniquePaths(int m, int n) {
		int[][] dp=new int[m][n];
		for(int i=0;i<m;i++)
			dp[i][0]=1;
		for(int i=0;i<n;i++)
			dp[m-1][0]=1;
		for(int i=m-2;i>=0;i--){
			for(int j=1;j<n;j++){
				dp[i][j]=dp[i][j-1]+dp[i+1][j];
			}
		}
		return dp[0][n-1];
	}
	
//	Write a method to determine if a string is a palindrome.   View Answer
	public boolean isPalindrome(String s) {
		if(s.length()<2)
			return true;
		s=s.toLowerCase();
		int i=0;
		int j=s.length()-1;
		while(i<j){
			while(i<j&&!Character.isLetterOrDigit(s.charAt(i)))
				i++;
			while(i<j&&!Character.isLetterOrDigit(s.charAt(j)))
				j--;
			if(s.charAt(i)!=s.charAt(j))
				return false;
			i++;
			j--;
		}
		return true;
	}
	
//	pow()
	public double pow(double x, int n) {
		if(n==0)
			return 1;
		boolean neg=n<0?true:false;
		n=neg?-n:n;
		double res=pow(x,n/2);
		if(n%2==0)
			res*=res;
		else
			res*=res*x;
		return neg?1/res:res;
	}
	
//	Fibonacci: recursive and iterative. 
	
	public int fibonacciRecur(int n){
		if(n==1)
			return 1;
		else if(n==2)
			return 1;
		else
			return fibonacciRecur(n-1)+fibonacciRecur(n-2);
	}
	
	public int fibonacci(int n){
		if(n==1)
			return 1;
		if(n==2)
			return 1;
		int last=1;
		int res=1;
		int total=2;
		for(int i=3;i<=n;i++){
			total=last+res;
			last=res;
			res=total;
		}
		return res;
	}
	
//	merge sorted list
	public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
		if(l1==null||l2==null)
			return l1==null?l2:l1;
		ListNode dummy=new ListNode(0);
		ListNode pre=dummy;
		while(l1!=null&&l2!=null){
			if(l1.val<l2.val){
				pre.next=l1;
				l1=l1.next;
			}
			else{
				pre.next=l2;
				l2=l2.next;
			}
			pre=pre.next;
		}
		if(l1!=null)
			pre.next=l1;
		if(l2!=null)
			pre.next=l2;
		return dummy.next;
	}
	
	public ListNode mergeTwoLists2(ListNode l1, ListNode l2) {
		if(l1==null||l2==null)
			return l1==null?l2:l1;
		if(l1.val<l2.val){
			l1.next=mergeTwoLists2(l1.next,l2);
			return l1;
		}
		else{
			l2.next=mergeTwoLists2(l1,l2.next);
			return  l2;
		}
	}
	
//	count and say
	public String countAndSay(int n) {
		if(n==1)
			return "1";
		String res="1";
		for(int i=1;i<n;i++){
			String temp="";
			char c=res.charAt(0);
			int count=1;
			for(int j=1;j<res.length();j++){
				if(res.charAt(j)==c)
					count++;
				else{
					temp=temp+count+c;
					c=res.charAt(j);
					count=1;
				}
			}
			temp=temp+count+c;
			res=temp;
		}
		return res;
	}
	
	
//	Given a number n, return a number formed from the same digits of n that is the number right before n. 
//	Example: Given 1342, you must return the number 1324.
	
	public void getNextSmaller(int[] num){
		int index=-1;
		for(int i=0;i<num.length-1;i++){
			if(num[i]>num[i+1])
				index=i;
		}

		if(index==-1)
			return;
		int idx=index+1;
		for(int i=index+1;i<num.length;i++){
			if(num[i]<num[index]){
				idx=i;
			}
		}
		
		int t=num[index];
		num[index]=num[idx];
		num[idx]=t;
		
		int beg=index+1;
		int end=num.length-1;
		while(beg<end){
			int tmp=num[beg];
			num[beg]=num[end];
			num[end]=tmp;
			beg++;
			end--;
		}
	}
	
	
	
//	 unix ls command--->simplify path
	public String simplifyPath(String path) {
		if(path.length()==0)
			return "/";
		String[] strs=path.split("/");
		Stack<String> stk=new Stack<String>();
		for(int i=0;i<strs.length;i++){
			String s=strs[i];
			if(s.equals(".")||s.isEmpty())
				continue;
			else if(s.equals("..")){
				if(!stk.isEmpty())
					stk.pop();
			}
			else
				stk.push(s);
		}
		if(stk.isEmpty())
			return "/";
		String res="";
		while(!stk.isEmpty()){
			res="/"+stk.pop()+res;
		}
		return res;
	}
	
//	multiply two strings: "123 * "45"
	public String multiply(String num1, String num2) {
		int l1=num1.length();
		int l2=num2.length();
		int[] res=new int[l1+l2];
		
		for(int i=l1-1;i>=0;i--){
			int carry=0;
			for(int j=l2-1;j>=0;j--){
				int prod=(num1.charAt(i)-'0')*(num2.charAt(j)-'0')+carry+res[i+j+1];
				carry=prod/10;
				prod=prod%10;
				res[i+j+1]=prod;
			}
			res[i]=carry;
		}
		
		String ans="";
		int k=0;
		while(k<res.length-1&&res[k]==0)// nice
			k++;
		while(k<res.length)
			ans=ans+res[k++];
		return ans;
	}

	
//	Check if two strings (including caps, whitespace, punctuation) are palindromes
	public boolean panlindromes(String s1, String s2){
		s1=s1.toLowerCase();
		s2=s2.toLowerCase();
		int i=0;
		int j=s2.length()-1;
		while(i<s1.length()&&j>=0){
			while(i<s1.length()&&!Character.isLetterOrDigit(s1.charAt(i)))
				i++;
			while(j>=0&&!Character.isLetterOrDigit(s2.charAt(j)))
				j--;
			if(s1.charAt(i)!=s2.charAt(j))
				return false;
			else{
				i++;
				j--;
			}
		}
		while(i<s1.length()){
			if(Character.isLetterOrDigit(s1.charAt(i)))
				return false;
			else
				i++;
		}
		
		while(j>=0){
			if(Character.isLetterOrDigit(s2.charAt(j)))
				return false;
			else
				j--;
		}
		return true;
	}
	
//	K nearest points to the origin on a 2D plane; 
	public List<Point> getKNearestPoints(Point [] points, int k) {
		Comparator<Point> comp=new Comparator<Point>(){

			@Override
			public int compare(Point o1, Point o2) {
				// TODO Auto-generated method stub
				if(o2.distanceFromOrigin()-o1.distanceFromOrigin()<0)
					return -1;
				else if(o2.distanceFromOrigin()-o1.distanceFromOrigin()>0)
					return 1;
				else
					return 0;
			}
			
		};
		PriorityQueue<Point> heap=new PriorityQueue<Point>(k, comp);
		for(Point p: points){
			if(heap.size()<k)
				heap.add(p);
			else{
				if(p.distanceFromOrigin()<heap.peek().distanceFromOrigin()){
					heap.poll();
					heap.add(p);
				}
			}
		}
		
		List<Point> res=new ArrayList<Point>();
		while(!heap.isEmpty()){
			res.add(heap.poll());
		}
		return res;
	}
	
//	Given a list of ranges, find whether the target range is in the union of the given intervals.
//	e.g: Input: a list of intervals, e.g. [-10, 10], [50, 100], [0, 20]
//	                 & a target range
//	    Output: true if target can be covered by the union of all intervals
//	       e.g. return true if target is [-5, 15]
	
	public boolean covered(List<Interval> intervals, Interval target){
		if(intervals.size()==0)
			return false;
		Comparator<Interval> comp=new Comparator<Interval>(){

			@Override
			public int compare(Interval o1, Interval o2) {
				// TODO Auto-generated method stub
				return o1.start-o2.start;
			}			
		};
		Collections.sort(intervals, comp);
		List<Interval> mergedList=new ArrayList<Interval>();
		mergedList.add(intervals.get(0));
		for(int i=1;i<intervals.size();i++){
			Interval cur=intervals.get(i);
			Interval last=mergedList.get(mergedList.size()-1);
			if(cur.start>last.end)
				mergedList.add(cur);
			else
				last.end=Math.max(cur.end, last.end);
		}

		for(int i=0;i<mergedList.size();i++){
			Interval interval=mergedList.get(i);
			if(interval.start<=target.start&&interval.end>=target.end)
				return true;
		}
		return false;
	}
	
	
//	search for range
	public int[] searchRange(int[] A, int target) {
		int[] res={-1,-1};
		int beg=0;
		int end=A.length-1;
		int index=-1;
		while(beg<=end){
			int mid=(beg+end)/2;
			if(A[mid]==target){
				index=mid;
				break;
			}
			else if(A[mid]>target)
				end=mid-1;
			else
				beg=beg+1;
		}
		if(index==-1)
			return res;
		beg=0;
		end=index;
		
		while(beg<=end){
			int mid=(beg+end)/2;
			if(A[mid]==target)
				end=mid-1;
			else
				beg=mid+1;
		}
		res[0]=beg;
		
		beg=index;
		end=A.length-1;
		while(beg<=end){
			int mid=(beg+end)/2;
			if(A[mid]==target)
				beg=mid+1;
			else
				end=mid-1;
		}
		res[1]=end;
		return res;
	}
	
	public void setZeroes(int[][] matrix) {
		int m=matrix.length;
		int n=matrix[0].length;
		int[] row=new int[m];
		int[] col=new int[n];
		
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++){
				if(matrix[i][j]==0){
					row[i]=1;
					col[j]=1;
				}
			}
		}
		
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++){
				if(row[i]==1||col[j]==1)
					matrix[i][j]=0;
			}
		}
	}
	
//	all path from root to leaf
	public List<List<Integer>> pathToLeaf(TreeNode root){
		List<List<Integer>> res= new ArrayList<List<Integer>>();
		List<Integer> path=new ArrayList<Integer>();
		List<Integer> offset=new ArrayList<Integer>();
		pathToLeafUtil(root, path, offset,0,res);
		
		for(int i=0;i<res.size();i+=2){
			List<Integer> nums=res.get(i);
			List<Integer> offsets=res.get(i+1);
			int min=0;
			for(int j=0;j<offsets.size();j++){
				min=Math.min(min, offsets.get(j));
			}
			for(int j=0;j<offsets.size();j++){
				int off=offsets.get(j);
				if(off<0){
					for(int k=0;k<off-min;k++)
						System.out.print(" ");
					System.out.println(nums.get(j));
				}
				else if(off==0){
					for(int k=0;k<0-min;k++)
						System.out.print(" ");
					System.out.println(nums.get(j));
				}
				else{
					
					for(int k=0;k<off-min;k++)
						System.out.print(" ");
					System.out.println(nums.get(j));
				}
				
			}
			System.out.println();
		}
		return res;
	}
	
	public void pathToLeafUtil(TreeNode root, List<Integer> path, List<Integer> offset, int curoff, List<List<Integer>> res){
		if(root==null)
			return;
		path.add(root.val);
		offset.add(curoff);
		if(root.left==null&&root.right==null){
			List<Integer> out=new ArrayList<Integer>(path);
			res.add(out);
			List<Integer> pos=new ArrayList<Integer>(offset);
			res.add(pos);
		}
		pathToLeafUtil(root.left,path,offset, curoff-1,res);
		pathToLeafUtil(root.right, path,offset, curoff+1, res);
		path.remove(path.size()-1);
		offset.remove(offset.size()-1);
	}
	
	public void printAllRootToLeafPaths(TreeNode node, ArrayList<Integer> path) 
	{
	    if(node==null)
	    {
	        return;
	    }
	    path.add(node.val);

	    if(node.left==null && node.right==null)
	    {
	        System.out.println(path);
	        return;
	    }
	    else
	    {
	        printAllRootToLeafPaths(node.left,new ArrayList<Integer>(path));
	        printAllRootToLeafPaths(node.right,new ArrayList<Integer>(path));
	    }      
	}
	
	
	public static int countPalindromeDP1(String s){
		int[][] dp=new int[s.length()][s.length()];
		for(int i=0;i<s.length();i++)
			dp[i][i]=1;
		
		for(int i=0;i<s.length()-1;i++){
			if(s.charAt(i)==s.charAt(i+1))
				dp[i][i+1]=1;
		}
		for(int k=3;k<=s.length();k++){
			for(int i=0;i<s.length()-k+1;i++){
				int j=i+k-1;
				if(dp[i+1][j-1]==1&&s.charAt(i)==s.charAt(j))
					dp[i][j]=1;
			}
		}
		int count=0;
		for(int i=0;i<s.length();i++){
			for(int j=0;j<s.length();j++){
				count+=dp[i][j];
			}
		}
		return count;
	}
	
	public static int countPalindromeDP2(String s){
		int n=s.length();
		int[][] dp=new int[n][n];
		int count=0;
		for(int i=n-1;i>=0;i--){
			for(int j=i;j<n;j++){
				if(i==j)
					dp[i][j]=1;
				else if(j==i+1){
					if(s.charAt(i)==s.charAt(j))
						dp[i][j]=1;
				}
				else{
					if(dp[i+1][j-1]==1&&s.charAt(i)==s.charAt(j))
						dp[i][j]=1;
				}
//				if(dp[i][j]==1)
//					System.out.println("start is "+i+", end is "+j);
				count+=dp[i][j];
			}
		}
		return count;
	}
	
	public static int countPalindrome2(String s){
		int count=s.length();
		int n=s.length();
		for(int i=0;i<n;i++){
			int j=i-1;
			int k=i+1;
			while(j>=0&&k<n&&s.charAt(j)==s.charAt(k)){
				count++;
				j--;
				k++;
			}
			
			j=i;
			k=i+1;
			while(j>=0&&k<n&&s.charAt(j)==s.charAt(k)){
				count++;
				j--;
				k++;
			}
		}
		return count;
	}
	// rearragne non-zero elements to the left of the array, zeros to the right
	public static void reArrange(int[] A){
		int i=0;
		int j=A.length-1;
		while(i<j){
			while(A[i]!=0)
				i++;
			while(A[j]==0)
				j--;
			if(i<j){
				int t=A[i];
				A[i]=A[j];
				A[j]=t;
			}
		}
	}
	
	public static void reArrange2(int[] A){
		int j=0;
		for(int i=0;i<A.length;i++){
			if(A[i]!=0)
				A[j++]=A[i];
		}
		while(j<A.length){
			A[j++]=0;
		}
	}
	
	//iterative 
//	Power set P(S) of a set S is the set of all subsets of S.
	public static List<List<Integer>> powerSet(int[] s){
		List<List<Integer>> res=new ArrayList<List<Integer>>();
		int n=s.length;
		int setSize=(int) Math.pow(2, n);
		for(int i=0;i<setSize;i++){
			List<Integer> sol=new ArrayList<Integer>();
			for(int j=0;j<n;j++){
				if((i&1<<j)!=0)
					sol.add(s[j]);
			}
			res.add(sol);
		}
		return res;
	}
	
	//recursive
	public static List<List<Integer>> powerSet2(int[] s){
		List<List<Integer>> res=new ArrayList<List<Integer>>();
		List<Integer> sol=new ArrayList<Integer>();
		powerSetUtil(0,s,sol,res);
		return res;
	}
	
	public static void powerSetUtil(int cur, int[] s, List<Integer> sol, List<List<Integer>> res){
		res.add(sol);
		if(cur==s.length)
			return;
		for(int i=cur;i<s.length;i++){
			List<Integer> out=new ArrayList<Integer>(sol);
			out.add(s[i]);
			powerSetUtil(i+1,s,out,res);
		}
	}
	
//	 给一个矩阵A，要求输出一个矩阵B，要求B[i,j]=sum(A[l,k]) l<=i,k<=j;
	
	public static int[][] transformMatrix(int[][] A){
		int m=A.length;
		int n=A[0].length;
		int[][] B= new int[m][n];
		for(int i=0;i<m;i++){
			for(int j=0;j<n;j++){
				int sum=0;
				for(int k=i;k>=0;k--){
					for(int l=j;l>=0;l--){
						sum+=A[k][l];
					}
				}
				B[i][j]=sum;
			}
		}
		return B;
	}
	// optimize the solution:Summed area table
	public static int[][] transformMatrix2(int[][] A){
		int m=A.length;
		int n=A[0].length;
		int[][] B= new int[m][n];
		B[0][0]=A[0][0];
		for(int i=1;i<m;i++){
			B[i][0]=B[i-1][0]+A[i][0];
		}
		for(int i=1;i<n;i++)
			B[0][i]=B[0][i-1]+A[0][i];
		
		for(int i=1;i<m;i++){
			for(int j=1;j<n;j++){
				B[i][j]=B[i-1][j]+B[i][j-1]-B[i-1][j-1]+A[i][j];
			}
		}
		return B;
	}

	
	
//  Longest Palindromic Substring Part II
	public String longestPalindrome(String s) {
		if(s.length()<2)
			return s;
		int start=0;
		int end=0;
		int max=1;
		for(int i=1;i<s.length();i++){
			int j=i-1;
			int k=i+1;
			while(j>=0&&k<s.length()&&s.charAt(j)==s.charAt(k)){
				j--;
				k++;
			}
			if(k-j+1>max){
				max=k-j+1;
				start=j+1;
				end=k-1;
			}
			
			j=i-1;
			k=i;
			while(j>=0&&k<s.length()&&s.charAt(j)==s.charAt(k)){
				j--;
				k++;
			}
			if(k-j+1>max){
				max=k-j+1;
				start=j+1;
				end=k-1;
			}
		}
		return s.substring(start, end+1);
	}
	
	
//	Given two strings representing integer numbers ("123" , "30") 
//	return a string representing the sum of the two numbers ("153")
	public String addNumbers(String s1, String s2){
		if(s1.isEmpty()||s2.isEmpty())
			return s1.isEmpty()?s2:s1;
		int i=s1.length()-1;
		int j=s2.length()-1;
		String res="";
		int carry=0;
		while(i>=0&&j>=0){
			int sum=s1.charAt(i)-'0'+s2.charAt(j)-'0'+carry;
			carry=sum/10;
			sum=sum%10;
			res=sum+res;
			i--;
			j--;
		}
		while(i>=0){
			int sum=s1.charAt(i)-'0'+carry;
			carry=sum/10;
			sum=sum%10;
			res=sum+res;
			i--;
		}
		
		while(j>=0){
			int sum=s2.charAt(j)-'0'+carry;
			carry=sum/10;
			sum=sum%10;
			res=sum+res;
			j--;
		}
		if(carry==1)
			return 1+res;
		return res;
		
	}
	
	
//	if a binary tree was valid 
	 public boolean isValidBST(TreeNode root) {
		 if(root==null)
			 return true;
		 return isValidBSTUtil(root, Integer.MIN_VALUE, Integer.MAX_VALUE);
	 }
	 public boolean isValidBSTUtil(TreeNode root, int leftmost, int rightmost){
		 if(root==null)
			 return true;
		 if(root.val<=leftmost||root.val>=rightmost)
			 return false;
		 return isValidBSTUtil(root.left,leftmost, root.val)&&isValidBSTUtil(root.right,root.val,rightmost);
	 }
	 
	// determine the minimum number of meeting rooms needed to hold all the meetings.
	// Input array(pair(1, 4), pair(2,3), pair(3,4), pair(4,5))
	// Output: 2
	 
//	 Given a list of interval , find the maximum overlaping. 
//	 For ex if input is 0,5 2,9 8,10 6,9 then ans is 2 as 8,10 overlap in 2,9 and 6,9
	 
	 public int maxIntervalOverlapping(List<Interval> intervals){
		 if(intervals.size()<2)
			 return 0;
		 int[] starts=new int[intervals.size()];
		 int[] ends=new int[intervals.size()];
		 for(int i=0;i<intervals.size();i++){
			 starts[i]=intervals.get(i).start;
			 ends[i]=intervals.get(i).end;
		 }
		 Arrays.sort(starts);
		 Arrays.sort(ends);
		 
		 int i=0;
		 int j=0;
		 int max=0;
		 
		 while(i<starts.length){
			 max++;
			 if(ends[j]<starts[i]){
				 max--;
				 j++;
			 }
			 i++;
		 }
		 return max;
	 }
	 
//		Minimum Number of Platforms Required for a Railway/Bus Station
		
		public int findPlatform(int arr[], int dep[]){
			Arrays.sort(arr);
			Arrays.sort(dep);
			
			int max=0;
			int platforms=1;
			int i=1;
			int j=0;
			while(i<arr.length && j<dep.length){
				if(arr[i]<dep[j]){
					platforms++;
					max=max<platforms?platforms:max;
					i++;
				}
				else{
					platforms--;
					j++;
				}
			}
			return max;
		}
	 
//	 find all valid paranthesis strings of length "2n" given an integer "n"
	 public List<String> generateParenthesis(int n) {
		 List<String> res=new ArrayList<String>();
		 generateParenthesisUtil(0,0,n,"",res);
		 return res;
	 }
	 
	 public void generateParenthesisUtil(int left, int right, int n, String sol,  List<String> res){
		 if(left==right&&left==n){
			 res.add(sol);
		 }
		 if(left<n)
			 generateParenthesisUtil(left+1,right,n,sol+"(",res);
		 if(right<left)
			 generateParenthesisUtil(left,right+1,n,sol+")",res);
		 
	 }
	 
	// maximum path in tree
	 
	 public int maxPathSum(TreeNode root) {
		 if(root==null)
			 return 0;
		 int[] res={Integer.MIN_VALUE};
		 maxPathSumUtil(root,res);
		 return res[0];
	 }
	 
	 public int maxPathSumUtil(TreeNode root, int[] res){
		 if(root==null)
			 return 0;
		 int left=maxPathSumUtil(root.left,res);
		 int right=maxPathSumUtil(root.right,res);
		 int single=Math.max(root.val, Math.max(left, right)+root.val);
		 int arch=left+right+root.val;
		 res[0]=Math.max(res[0],Math.max(single, arch) );
		 return single;
	 }
	 
	 
//	 Given a set of n jobs with [start time, end time, cost] find a subset
//	 so that no 2 jobs overlap and the cost is maximum ?
	 public List<Interval> maxSubsetNoOverlapping(List<Interval> intervals){
		 if(intervals.size()<2)
			 return intervals;
		 Comparator<Interval> comp=new Comparator<Interval>(){

			@Override
			public int compare(Interval o1, Interval o2) {
				// TODO Auto-generated method stub
				return o1.end-o2.end;
			}
			 
		 };
		 List<Interval> res=new ArrayList<Interval>();
		 Collections.sort(intervals, comp);
		 for(int i=0;i<intervals.size();i++){
			 Interval interval=intervals.get(i);
			 if(res.size()==0)
				 res.add(interval);
			 else{
				 if(interval.start>=res.get(res.size()-1).end)
					 res.add(interval);
			 }
		 }
		 return res;
	 }
	 
//		Given a bipartite graph, separate the vertices into two sets.
	 public int[] separateBipartite(int[][] matrix){
		 int n=matrix.length;
		 int[] color=new int[n];
		 for(int i=0;i<n;i++)
			 color[i]=-1;
		 
		 Queue<Integer> que=new LinkedList<Integer>();
		 color[0]=1;
		 que.add(0);
		 
		 while(!que.isEmpty()){
			 int u=que.remove();
			 for(int v=0;v<n;v++){
				 if(matrix[u][v]==1&&color[v]==-1){
					 color[v]=1-color[u];
					 que.add(v);
				 }
			 }
		 }
		 return color;
	 }
	 
//	 a) first, write a function to calculate the hamming distance between two binary numbers
	 public int hammingDist(String n1, String n2) {
		 String small=n1.length()<n2.length()?n1:n2;
		 String large=n1.length()<n2.length()?n2:n1;
		 int dif=large.length()-small.length();
		 for(int i=0;i<dif;i++){
			 small="0"+small;
		 }
		 int res=0;
		 for(int i=small.length()-1;i>=0;i--){
			 if(small.charAt(i)!=large.charAt(i))
				 res++;
		 }
		 return res;
	 }
	 
//	 (b) write a function that takes a list of binary numbers and returns the sum of the hamming distances for each pair
//
//	 (c) find a solution for (b) that works in O(n) time.
	 
	 int hammingDist(List<Integer> nums) {
		 
			int[] bits=new int[32];
		 
			for(int i = 0; i < nums.size(); i++) {
				int one = 1;
				int j = 0;
				int num=nums.get(i);
				while(num>0) {
					if((num&one)==1) 
						bits[j]++;// count totally how many 1s
					j++;
					num>>=1;
				}
			}
			
			int res = 0;
			for(int i = 0; i < 32; i++) {
				// nums.size()-bits[i] means how many 0s, for each bit changing from 1 to 0, it needs no.of 1s * no. of 0s
				res += bits[i]*(nums.size()-bits[i]); 
			}
			
			return res;
		}
	 
	 public int minDistance(String word1, String word2) {
		 int m=word1.length();
		 int n=word2.length();
		 int[][] edits=new int[m+1][n+1];
		 for(int i=0;i<=m;i++)
			 edits[i][0]=i;
		 for(int i=1;i<=n;i++)
			 edits[0][i]=i;
		 
		 for(int i=1;i<=m;i++){
			 for(int j=1;j<=n;j++){
				 if(word1.charAt(i-1)==word2.charAt(j-1))
					 edits[i][j]=edits[i-1][j-1];
				 else
					 edits[i][j]=Math.min(Math.min(edits[i-1][j], edits[i][j-1]),edits[i-1][j-1])+1;
			 }
		 }
		 return edits[m][n];
	 }
	 
//	 Print all permutation of a given string. 
	 public List<String> stringPermutation(String s){
		 List<String> res = new ArrayList<String>();
		 if(s.length()==0){
			 res.add("");
			 return res;
		 }
		 char c=s.charAt(0);
		 List<String> perms=stringPermutation(s.substring(1));
		 
		 for(int i=0;i<perms.size();i++){
			 String word=perms.get(i);
			 for(int j=0;j<=word.length();j++){
				 res.add(word.substring(0,j)+c+word.substring(j));
			 }
		 }
		 return res;
	 }
	 
	 public List<String> stringPermutation2(String s){
		 List<String> res = new ArrayList<String>();
		 boolean[] used=new boolean[s.length()];
		 stringPermutation(0,s,used,"",res);
		 return res;
	 }
	 
	 public void stringPermutation(int cur, String s, boolean[] used, String sol, List<String> res){
		 if(cur==s.length()){
			 res.add(sol);
			 return;
		 }
		 for(int i=0;i<s.length();i++){
			 if(!used[i]){
				 used[i]=true;
				 stringPermutation(cur+1,s,used, sol+s.charAt(i),res);
				 used[i]=false;
			 }
			 
		 }
	 }
	 
	 public void bubbleSort(int[] A){
		 if(A.length<2)
			 return;
		 boolean flag=true;
		 
		 while(flag){
			 flag=false;
			 for(int i=0;i<A.length-1;i++){
				 if(A[i]>A[i+1]){// change to < for descending sort
					 int t=A[i];
					 A[i]=A[i+1];
					 A[i+1]=t;
					 flag=true;
				 }
			 }
		 }
	 }
//finding the number of ways a given score could be reached for a game with 3 different ways of scoring (e.g. 3, 5 and 10 points)
	 public List<List<Integer>> combinationSum(int[] candidates, int target) {
		 List<List<Integer>> res=new ArrayList<List<Integer>>();
		 Arrays.sort(candidates);
		 List<Integer> sol=new ArrayList<Integer>();
		 combinationSum(0,candidates,sol, 0, target, res);
		 return res;
	 }
	 
	 public void combinationSum(int dep, int[] candidates, List<Integer> sol, int cursum, int target, List<List<Integer>> res){
		 if(dep==candidates.length||cursum>target)
			 return;
		 if(cursum==target){
			 List<Integer> out=new ArrayList<Integer>(sol);
			 res.add(out);
		 }
		 
		 for(int i=dep;i<candidates.length;i++){
			 cursum+=candidates[i];
			 sol.add(candidates[i]);
			 combinationSum(i+1,candidates, sol,cursum,target,res);
			 cursum-=candidates[i];
			 sol.remove(sol.size()-1);
		 }
		 
	 }
	 
	//make coin change
	 
	 public int coinChange(int[] coins, int target){
		 int n=coins.length;
		 int[][] table=new int[n][target+1];
		 
		 for(int i=0;i<n;i++)
			 table[i][0]=1;
		 for(int i=0;i<n;i++){
			 for(int j=1;j<target+1;j++){
				// Count of solutions including coin[i]
				 int x=j>=coins[i]?table[i][j-coins[i]]:0;
				// Count of solutions excluding coin[i]
				 int y=i>0?table[i-1][j]:0;
				 table[i][j]=x+y;
			 }
		 }
		 return table[n-1][target];
	 }
	 
	 public int coinChange2(int[] coins, int target){
		 int n=coins.length;
		 int[] table=new int[target+1];
		 table[0]=1;
		 
		 for(int i=0;i<n;i++){
			 for(int j=0;j<target+1;j++){
//				 table[j]=table[j-1];
				 if(j>=coins[i])
					 table[j]+=table[j-coins[i]];
			 }
		 }
		 return table[target];
	 }
	 
	 public int getHeight(TreeNode root){
		 if(root==null)
			 return 0;
		 int left=getHeight(root.left);
		 int right=getHeight(root.right);
		 return left>right?left+1:right+1;
	 }

//	 find average value for each level in binary tree. 
	 public void levelAverage(TreeNode root){
		 if(root==null)
			 return;
		 int h=getHeight(root);
		 for(int i=1;i<=h;i++){
			 int[] sum={0};
			 int[] count={0};
			 levelAverage(root, i, sum, count);
			 System.out.println(sum[0]/(count[0]));
		 }
		 
	 }
	 
	 public void levelAverage(TreeNode root, int level, int[] sum, int[] count){
		 if(root==null)
			 return;
		 if(level==1){
			 sum[0]+=root.val;
			 count[0]++;
		 }
		 else{
			 levelAverage(root.left,level-1,sum,count);
			 levelAverage(root.right,level-1,sum,count);
		 }
		 
	 }
	 
	 public void levelAverage2(TreeNode root){
		 if(root==null)
			 return;
		 HashMap<Integer, List<Integer>> map=new HashMap<Integer, List<Integer>>();
		 levelAverageUtil(root, 0, map);
		 Iterator<Integer> it=map.keySet().iterator();
		 while(it.hasNext()){
			 int level=it.next();
			 List<Integer> lst=map.get(level);
			 int sum=0;
			 for(int i:lst)
				 sum+=i;
			 System.out.println("Average of Levle "+level+" is "+sum/lst.size());
		 }
	 }
	 public void levelAverageUtil(TreeNode root, int level, HashMap<Integer, List<Integer>> map){
		 if(root==null)
			 return;
		 if(!map.containsKey(level)){
			 List<Integer> lst=new ArrayList<Integer>();
			 lst.add(root.val);
			 map.put(level, lst);
		 }
		 else{
			 map.get(level).add(root.val);
		 }
		 levelAverageUtil(root.left,level+1,map);
		 levelAverageUtil(root.right,level+1,map);
	 }
	 
//	 给你一个double func(double x)，你能调用这个函数然后它会返回一个值，要求实现一个double invert(double y, double start, double end)。
//	 保证func在区间（start， end）上是单调增的，要求返回一个x使得func(x) = y。
	
	 
	 public double func(double x) {
		 return Math.pow(x, 2);
	 }
	 
	 public double invert(double y, double start, double end){
		 while(start<=end){
			 double mid=(start+end)/2.0;
			 if(Math.abs(func(mid)-y)<0.0000001)
				 return mid;
			 else if(func(mid)>y)
				 end=mid;
			 else
				 start=mid;
		 }
		 return -1;
	 }
	 
//	 Write a function that computes log2() using sqrt().
	 public int log2(int num) {
			
			if(num <= 2) return 1;
			
			int s = sqrt(num);
			if(s*s == num) 
				return 2*log2(s);
			else 
				return 1+log2(num/2);
		}
	 
//	 Implement a power function to raise a double to an int power, including negative powers.
	 
	 public double power(double x, int n){
		 if(n==0)
			 return 1;
		 boolean neg=n<0?true:false;
		 if(neg)
			 n=-n;
		 double res=power(x,n/2);
		 if(n%2==0)
			 res*=res;
		 else
			 res*=res*x;
		 return neg?1.0/res:res;
	 }
	 
	 public double iter_power(double x, int n){
		 if(n==0)
			 return 1;
		 boolean neg=n<0?true:false;
		 if(neg)
			 n=-n;
		 double res=1;
		 while(n>0){
			 if(n%2!=0)
				 res*=x;
			 x*=x;
			 n/=2;			 
		 }
		 return neg?1.0/res:res;
	 }
		
//		longest common subsequence 
	 public int longestCommonSubsequence(String A, String B) {
	        // write your code here
	        int m=A.length();
	        int n=B.length();
	        int[][] dp=new int[m+1][n+1];
	        for(int i=1;i<=m;i++){
	            for(int j=1;j<=n;j++){
	                if(A.charAt(i-1)==B.charAt(j-1))
	                    dp[i][j]=dp[i-1][j-1]+1;
	                else
	                    dp[i][j]=Math.max(dp[i-1][j],dp[i][j-1]);
	            }
	        }
	        return dp[m][n];
	    }
	 public double sqrt(double x){
		 double i=0;
		 double j=x;
		 while(Math.abs(i-j)>=1e-9){
			 double mid=(i+j)/2;
			 if(mid*mid<x)
				 i=mid;
			 else
				 j=mid;
		 }
		 return i;
	 }
	 
//		how many combinations you can make out of a set of numbers.
	 public List<List<Integer>> combine(int n, int k) {
		 List<List<Integer>> res=new ArrayList<List<Integer>>();
		 List<Integer> sol=new ArrayList<Integer>();
		 combineUtil(0, k,n,sol,res,1);
		 return res;
	 }
	 
	 public void combineUtil(int dep, int maxDep, int n, List<Integer> sol, List<List<Integer>> res, int curpos){
		 if(dep==maxDep){
			 List<Integer> out=new ArrayList<Integer>(sol);
			 res.add(out);
		 }
		 for(int i=curpos;i<=n;i++){
			 sol.add(i);
			 combineUtil(dep+1,maxDep, n,sol,res,i+1);
			 sol.remove(sol.size()-1);
		 }
	 }
	 
//	  Insert a node in a singly linked circular list given any node in the list.
	 
	 public ListNode insert(ListNode head, int d){
		 if(head==null){
			 head=new ListNode(d);
			 head.next=head;
			 return head;
		 }
		 ListNode cur=head;
		 while(cur.next!=head)
			 cur=cur.next;
		 ListNode node=new ListNode(d);
		 cur.next=node;
		 node.next=head;
		 return head;
	 }
	 
	 public int maximumSum(int arr[]){
		 int n=arr.length;
		 if(n==0)
			 return 0;
		 int[] res=new int[arr.length];
		 res[0]=arr[0];
		 res[1]=arr[0]>arr[1]?arr[0]:arr[1];
		 for(int i=2;i<n;i++){
			 res[i]=Math.max(res[i-2]+arr[i], res[i-1]);
		 }
		 return res[n-1];
		
	 }

//	 Given a m*n grid starting from (1, 1). At any point (x, y), you have two choices for the next move:
//	     1) move to (x+y, y);
//	     2) move to (x, y+x);
//	From point (1, 1), how to move to (m, n) in least moves? (or there's no such a path).
	 public int minMoves(int x, int y) {
			
			int cnt = 0;
			while(true) {
				if(x == 1 && y == 1) 
					break;
				else if(x == y) 
					return -1; // gcd(x, y) != 1
				
				if(x > y) x -= y;
				else y -= x;
				
				cnt++;
			}
			
			return cnt;
		}
	 

	 
	 
	//flatten muti-level list
	public ComplexNode flattenList(ComplexNode head){// level order
		if(head==null)
			return null;
		ComplexNode tail=head;
		while(tail.next!=null){
			tail=tail.next;
		}
		ComplexNode cur=head;
		while(cur!=tail){
			if(cur.child!=null){
				tail.next=cur.child;
				ComplexNode node=cur.child;
				while(node.next!=null)
					node=node.next;
				tail=node;
			}
			cur=cur.next;
		}
		return head;
	}
	
	public ComplexNode flattenList2(ComplexNode head){//--no order
		if(head==null)
			return null;
		Queue<ComplexNode> que=new LinkedList<ComplexNode>();
		que.offer(head);
		ComplexNode dummy=new ComplexNode(0);
		ComplexNode pre=dummy;
		while(!que.isEmpty()){
			ComplexNode node=que.poll();
			pre.next=node;
			pre=pre.next;
			if(node.next!=null)
				que.offer(node.next);
			if(node.child!=null)
				que.offer(node.child);
		}
		return dummy.next;
	}
	
	// strStr
	public int strStr(String haystack, String needle) {
		for(int i=0;i<haystack.length()-needle.length()+1;i++){
			int j;
			for(j=0;j<needle.length();j++){
				if(haystack.charAt(i+j)!=needle.charAt(j))
					break;
			}
			if(j==needle.length())
				return i;
		}
		return -1;
	}
//	 anagrams
	 public List<List<String>> anagrams(String[] strs) {
		 List<List<String>> res=new ArrayList<List<String>>();
		 HashMap<String, List<String>> map=new HashMap<String, List<String>>();
		 
		 for(String s:strs){
			 char[] chars=s.toCharArray();
			 Arrays.sort(chars);
			 String str=new String(chars);
			 if(map.containsKey(str))
				 map.get(str).add(s);
			 else{
				 List<String> lst=new ArrayList<String>();
				 lst.add(s);
				 map.put(str, lst);
			 }
		 }
		 Iterator<String> it=map.keySet().iterator();
		 while(it.hasNext()){
			 String s=it.next();
			 res.add(map.get(s));
		 }
		 return res;
	 }
	 
	 
//	 Smallest missing natural number in a linked list in linear time without a hash table. 
	 public int firstMissing(ListNode head){
		 if(head==null)
			 return 1;
		 int len=0;
		 ListNode cur=head;
		 while(cur!=null){
			 len++;
			 cur=cur.next;
		 }
		 boolean[] A=new boolean[len];
		 cur=head;
		 
		 while(cur!=null){				 
			 if(cur.val>0&&cur.val<=len)
				 A[cur.val-1]=true;
			 cur=cur.next;
		 }
		 for(int i=0;i<len;i++)
			 if(!A[i])
				 return i+1;
		 return len+1;
	 }

	 
//	 Given an unsorted array, extract the max and min value using the least number of comparison. 
//	 the strategy is to go through the elements in pairs, and compare the smaller one to the minimum, and the bigger one to the maximum. This is 3 comparisons, done n/2 times in total, for 3n/2 running time.
	// 3*n/2
	 
	 void leastComparisonMinMax(int arr[]) {
		 int n=arr.length;
			int mn = arr[0], mx = arr[0];
			
			for(int i = 1; i < n;) {
				if(i+1 >= n) {
					mn = Math.min(mn, arr[i]);
					mx = Math.max(mx, arr[i]);
					i++;
				}
				else {
					if(arr[i] <= arr[i+1]) {
						mn = Math.min(arr[i], mn);
						mx = Math.max(arr[i+1], mx);
					}
					else {
						mn = Math.min(arr[i+1], mn);
						mx = Math.max(arr[i], mx);
					}
					
					i += 2;
				}
			}
			
			System.out.println(mn + ", " + mx);
		}
	

//paint-fill
//	Write a method to implement the flood fill algorithm.---BFS---DFS
//	Flood-fill (node, target-color, replacement-color):
//		 1. If target-color is equal to replacement-color, return.
//		 2. Set Q to the empty queue.
//		 3. Add node to the end of Q.
//		 4. While Q is not empty: 
//		 5.     Set n equal to the last element of Q.
//		 6.     Remove last element from Q.
//		 7.     If the color of n is equal to target-color:
//		 8.         Set the color of n to replacement-color and mark "n" as processed.
//		 9.         Add west node to end of Q if west has not been processed yet.
//		 10.        Add east node to end of Q if east has not been processed yet.
//		 11.        Add north node to end of Q if north has not been processed yet.
//		 12.        Add south node to end of Q if south has not been processed yet.
//		 13. Return.
	
	 public void floodfill(int[][] matrix, Point2 point, int target, int replacement){
		 if(target==replacement)
			 return;
		 int X[] = {0, 1, 1, 0, -1, 1, -1, -1};
		 int Y[] = {1, 0, 1, -1, 0, -1, 1, -1};
		 int m=matrix.length;
		 int n=matrix[0].length;
		 boolean[][] visited=new boolean[m][n];
		 Queue<Point2> que=new LinkedList<Point2>();
		 que.offer(point);
		 while(!que.isEmpty()){
			 Point2 p=que.poll();
			 visited[p.x][p.y]=true;
			 if(matrix[p.x][p.y]==replacement)
				 continue;
			 if(matrix[p.x][p.y]==target)
				 matrix[p.x][p.y]=replacement;
			 for(int i=0;i<8;i++){
				 if(X[i]+p.x >= 0 && X[i]+p.x < m && Y[i]+p.y >= 0 && Y[i]+p.y < n&&!visited[X[i]+p.x][Y[i]+p.y]) {
						Point2 np = new Point2(p.x+X[i], p.y+Y[i]);
						que.offer(np);
					}
			 }
//			 if(p.x+1<m&&!visited[p.x+1][p.y])
//				 que.offer(new Point2(p.x+1,p.y));
//			 if(p.x-1>=0&&!visited[p.x-1][p.y])
//				 que.offer(new Point2(p.x-1,p.y));
//			 if(p.y+1<n&&!visited[p.x][p.y+1])
//				 que.offer(new Point2(p.x,p.y+1));
//			 if(p.y-1>=0&&!visited[p.x][p.y-1])
//				 que.offer(new Point2(p.x,p.y-1));
		 }
	 }
	 
	//Recursive 4-way floodfill, crashes if recursion stack is full 
	 void floodFill4(int[][] screenBuffer, int x, int y, int oldColor, int newColor) 
	 { 
	     if(x >= 0 && x < screenBuffer.length && y >= 0 && y < screenBuffer[0].length && screenBuffer[x][y] == oldColor && screenBuffer[x][y] != newColor) 
	     { 
	         screenBuffer[x][y] = newColor; //set color before starting recursion
	         
	         floodFill4(screenBuffer,x + 1, y, oldColor, newColor);
	         floodFill4(screenBuffer,x - 1, y, oldColor, newColor);
	         floodFill4(screenBuffer,x, y + 1, oldColor, newColor);
	         floodFill4(screenBuffer,x, y - 1, oldColor, newColor);
	     }     
	 }

	//4-way floodfill using our own stack routines
	 void floodFill4Stack(int[][] screenBuffer, int x, int y, int newColor, int oldColor)
	 {
	     if(newColor == oldColor) return; //avoid infinite loop
	     Stack<Point2> stk=new Stack<Point2>();
	      Point2 point=new Point2(x,y);
	     stk.push(point);
	     int m=screenBuffer.length;
	     int n=screenBuffer[0].length;
	     boolean[][] visited=new boolean[m][n];
	     while(!stk.isEmpty())
	     {
	    	 Point2 p=stk.pop();
	    	 x=p.x;
	    	 y=p.y;
	    	 visited[x][y]=true;
	         screenBuffer[x][y] = newColor;
	         if(x + 1 < m && screenBuffer[x + 1][y] == oldColor&&!visited[x+1][y])
	         {          
	             stk.push(new Point2(x + 1, y));           
	         }    
	         if(x - 1 >= 0 && screenBuffer[x - 1][y] == oldColor&&!visited[x-1][y])
	         {
	        	 stk.push(new Point2(x - 1, y));            
	         }    
	         if(y + 1 < n && screenBuffer[x][y + 1] == oldColor&&!visited[x][y+1])
	         {
	        	 stk.push(new Point2(x, y+1));          
	         }    
	         if(y - 1 >= 0 && screenBuffer[x][y - 1] == oldColor&&!visited[x][y-1])
	         {
	        	 stk.push(new Point2(x, y-1));           
	         }    
	     }     
	 }

	public List<List<Integer>> levelTraversal(TreeLinkNode root){
		List<List<Integer>> res=new ArrayList<List<Integer>>();
		if(root==null)
			return res;
		TreeLinkNode cur=root;
		while(cur!=null){
			TreeLinkNode node=cur;
			List<Integer> level=new ArrayList<Integer>();
			TreeLinkNode next=null;
			while(node!=null){
				level.add(node.val);
				if(node.left!=null&&next==null)
					next=node.left;
				else if(node.right!=null&&next==null)
					next=node.right;
				node=node.next;
			}
			res.add(level);
			cur=next;
			
		}
		return res;
	}
	
	 public void connect(TreeLinkNode root) {
		 if(root==null)
			 return;
		 TreeLinkNode cur=root;
		 TreeLinkNode nextLvHead=null;
		 TreeLinkNode nextLvEnd=null;
		 while(cur!=null){
			 if(cur.left!=null){
				 if(nextLvHead==null){
					 nextLvHead=cur.left;
					 nextLvEnd=cur.left;
				 }
				 else{
					 nextLvEnd.next=cur.left;
					 nextLvEnd=cur.left;
				 }
			 }
			 
			 if(cur.right!=null){
				 if(nextLvHead==null){
					 nextLvHead=cur.right;
					 nextLvEnd=cur.right;
				 }
				 else{
					 nextLvEnd.next=cur.right;
					 nextLvEnd=cur.right;
				 }
			 }
			 cur=cur.next;
			 if(cur==null){
				 cur=nextLvHead;
				 nextLvHead=null;
				 nextLvEnd=null;
			 }
		 }
	 }
	 
//	 要先做右子树，再作左子树，考虑以下情况：
//	 1. l1和r1分别为root节点的两个子节点，如果说假设我们先做l1
//	 2. 做到l1的右子节点的时候，需要到r1的子节点里面去找next，这时候如果r1的两个子节点都是空，那么需要继续到r1的next中去找，这时候因为我们先递归了l1，r1的next还没有被赋值，所以会出现丢失next的情况。
	 
	 public void connect2(TreeLinkNode root) {
		if(root==null)
			return;
		if(root.left!=null)
			root.left.next=root.right==null?findNext(root.next):root.right;
			
		if(root.right!=null)
			root.right.next=findNext(root.next);
		
		connect2(root.right);
		connect2(root.left);
	 }
	 
	 public TreeLinkNode findNext(TreeLinkNode root){
		 if(root==null)
			 return null;
		 TreeLinkNode cur=root;
		 while(cur!=null){
			 if(cur.left!=null)
				 return cur.left;
			 if(cur.right!=null)
				 return cur.right;
			 cur=cur.next;
		 }
		 return null;
	 }
	 
//	 single number，数组有序，要log(n)
//	 行升序矩阵，找kth，分析多种方法
//	 flatten二叉树
//	 bfs遍历二叉树，从左到右返叶子节点
	 
	 public int singleNum(int[] A){
		 int i=0;
		 int j=A.length-1;
		 while(i<=j){
			 int mid=(i+j)/2;
			 System.out.println(mid);
			 if((mid==0||A[mid]>A[mid-1])&&(mid==A.length-1||A[mid]<A[mid+1]))
					 return A[mid];
			 else if((A[mid]==A[mid-1]&&mid%2==0)||(A[mid]==A[mid+1]&&mid%2==1))
				 j=mid-1;
			 else
				 i=mid+1;
		 }
		 return A[i];
	 }
	 
	 public List<Integer> getLeaves(TreeNode root){
		 List<Integer> res = new ArrayList<Integer>();
		 getLeaves(root, res);
		 return res;
	 }
	 public void getLeaves(TreeNode root, List<Integer> res){
		 if(root==null)
			 return ;
		 if(root.left==null&&root.right==null)
			 res.add(root.val);
		 getLeaves(root.left,res);
		 getLeaves(root.right,res);
		 
	 }
	 
	 /* Given a perfect binary tree, print its nodes in specific
	   level order left, right alternatively*/
	 public List<Integer> printSpecificLevelOrder(TreeNode root){
		 List<Integer> res=new ArrayList<Integer>();
		 if(root==null)
			 return res;
		 Queue<TreeNode> que=new LinkedList<TreeNode>();
		 que.offer(root);
		 TreeNode node1=null, node2=null;
		 while(!que.isEmpty()){
			 node1=que.poll();
			 res.add(node1.val);
			 if(!que.isEmpty()){
				 node2=que.poll();
				 res.add(node2.val);
			 }
				 
			 if(node1.left!=null){
				 que.offer(node1.left);
				 if(node2!=null)
					 que.offer(node2.right);
				 que.offer(node1.right);
				 if(node2!=null)
					 que.offer(node2.left);
			 }
		 }
		 return res;
	 }
	 
//	 You're given a dictionary of strings, and a key. 
//	 Check if the key is composed of an arbitrary number of concatenations of strings from the dictionary.  
	 
	 public boolean isConcantentationOfOtherStrings(Set<String> dict, String word){
		 int n=word.length();
		 boolean[] table=new boolean[n+1];
		 table[0]=true;
		 
		 for(int i=0;i<n;i++){
			 for(int j=i;j>=0;j--){
				 if(table[j]){
					 if(dict.contains(word.substring(j,i+1))){
						 table[i+1]=true;
						 break;
					 }
				 }
			 }
		 }
		 return table[n];
	 }
	 
	 public boolean isConcantentationOfOtherStrings2(Set<String> dict, String word){
		 int n=word.length();
		 if(word.length()==0)
			 return true;
		 for(int i=0;i<n;i++){
			 if(dict.contains(word.substring(0,i+1)))
				 if(isConcantentationOfOtherStrings2(dict,word.substring(i+1)))
					 return true;
		 }
		 return false;
	 }
	 
	 public RandomListNode copyRandomList(RandomListNode head) {
		 if(head==null)
			 return null;
		 RandomListNode cur=head;
		 
		 while(cur!=null){
			 RandomListNode next=cur.next;
			 RandomListNode copy=new RandomListNode(cur.label);
			 cur.next=copy;
			 copy.next=next;
			 cur=next;
		 }
		 
		 cur=head;
		 while(cur!=null){
			 if(cur.random!=null)
				 cur.next.random=cur.random.next;
			 cur=cur.next.next;
		 }
		 
		 cur=head;
		 RandomListNode copyHead=cur.next;
		 RandomListNode cur1=copyHead;
		 while(cur!=null){
			 cur.next=cur1.next;
			 if(cur.next!=null)
				 cur1.next=cur.next.next;
			 cur=cur.next;
			 cur1=cur1.next;
		 }
		 return copyHead;
	 }
	 
//	 You're given an array of integers(eg [3,4,7,1,2,9,8]) Find the index of values that satisfy A+B = C + D, where A,B,C & D are integers values in the array. 
//
//	 Eg: Given [3,4,7,1,2,9,8] array 
//	 The following 
//	 3+7 = 1+ 9 satisfies A+B=C+D 
//	 so print (0,2,3,5)
	 
	 public void indexSumPair(int[] arr){	
		 if(arr.length<4)
			 return;
		 Map<Integer, List<Pair>> map=new HashMap<Integer, List<Pair>>();
		 for(int i=0;i<arr.length-1;i++){
			 for(int j=i+1;j<arr.length;j++){
				 int sum=arr[i]+arr[j];
				 Pair p=new Pair(i,j);
				 if(map.containsKey(sum)){
					 map.get(sum).add(p);
				 }
				 else{
					 List<Pair> lst=new ArrayList<Pair>();
					 lst.add(p);
					 map.put(sum, lst);
				 }
			 }
		 }
		 Iterator<Integer> it=map.keySet().iterator();
		 while(it.hasNext()){
			 int key=it.next();
			 List<Pair> lst=map.get(key);
			 if(lst.size()>1){
				 for(int i=0;i<lst.size()-1;i++){
					 for(int j=i+1;j<lst.size();j++){
						 System.out.println(lst.get(i)+" = "+lst.get(j));
					 }
				 }
			 }
		 }
	 }
	 
	 
//	 Code a function that receives a string composed by words separated by spaces and returns a string where words appear in the same order but than the original string, but every word is inverted. 
//	 Example, for this input string
//
//
//	 @"the boy ran"
//	 the output would be
//
//
//	 @"eht yob nar"
	 
	 public  String reverse(String s){
		 if(s.length()<2)
			 return s;
		 
		 String res="";
		 int start=0;
		 for(int i=0;i<s.length();i++){
			 if(s.charAt(i)==' '){
				 res=reverse(res, start, i-1);
				 start=i+1;
			 }
			 res+=s.charAt(i);	 
		 }
		 res=reverse(res,start, s.length()-1);
		 return res;
	 }
	 
	 public String reverse(String s, int start, int end){
		 char[] chars=s.toCharArray();
		 while(start<end){
			 char c=chars[start];
			 chars[start]=chars[end];
			 chars[end]=c;
			 
			 start++;
			 end--;
		 }
		 return new String(chars);
	 }
	 
	 public boolean exist(char[][] board, String word) {
		 int m=board.length;
		 if(m==0)
			 return false;
		 int n=board[0].length;
		 boolean[][] visited=new boolean[m][n];
		 for(int i=0;i<m;i++){
			 for(int j=0;j<n;j++){
				 if(board[i][j]==word.charAt(0)){
					 if(dfs(i, j, board, 0, word, visited))
						 return true;
				 }
			 }
		 }
		 return false;
	 }
	 
	 public boolean dfs(int i, int j, char[][] board, int cur, String word, boolean[][] visited){
		 if(cur==word.length())
			 return true;
		 if(i>=0&&i<board.length&&j>=0&&j<board[0].length&&!visited[i][j]&&word.charAt(cur)==board[i][j]){
			 visited[i][j]=true;
			 boolean found=dfs(i+1,j,board,cur+1,word,visited)||
			 dfs(i-1,j,board,cur+1,word,visited)||
			 dfs(i,j+1,board,cur+1,word,visited)||
			 dfs(i,j-1,board,cur+1,word,visited);
			 if(found)
				 return true;
			 else
				 visited[i][j]=false;
		 }
		 return false;
	 }
	 
	 public void flatten(TreeNode root) {
		 if(root==null||root.left==null&&root.right==null)
			 return;
		 if(root.left!=null){
			 TreeNode right=root.right;
			 root.right=root.left;
			 
			 TreeNode cur=root.left;
			 while(cur.right!=null)
				 cur=cur.right;
			 cur.right=right;
			 root.left=null;
		 }
		 flatten(root.right);
		 
	 }
	 
	 public void flatten2(TreeNode root) {
		 if(root==null)
			 return;
		 TreeNode cur=root;
		 while(cur!=null){
			 if(cur.left!=null){
				 TreeNode left=cur.left;
				 while(left.right!=null)
					 left=left.right;
				 left.right=cur.right;
				 cur.right=cur.left;
				 cur.left=null;
			 }
			 cur=cur.right;
		 }
	 }
	 
	 public void flatten3(TreeNode root) {
		 if(root==null)
			 return;
		 TreeNode right=root.right;
		 TreeNode left=root.left;
		 root.right=null;
		 root.left=null;
		 flatten(left);
		 flatten(right);
		 if(left!=null)
			 root.right=left;
		 TreeNode node=root;
		 while(node.right!=null)
			 node=node.right;
		 node.right=right;
		 
	 }
	 
//	 （以Git bisect 为背景的题目， 他给解释了一下bisect是干什么的，没有听懂，不影响做题）
//	 题目抽象出来就是一个布尔型的数组，从某一项开始就全变成False了，找出第一个False的index。已知第一项一定为真，最后一项一定为假。
	 
	 public int firstFalse(boolean[] git){
		 int beg=0;
		 int end=git.length-1;
		 while(beg<=end){
			 int mid=(beg+end)/2;
			 if(mid>0&&git[mid-1]&&!git[mid]||mid==0&&!git[mid])
				 return mid;
			 if(git[mid])
				 beg=mid+1;
			 else
				 end=mid-1;
		 }
		 return -1;
	 }
	 
//	 Given an array of positive integers that represents possible points a team could score in an individual play. Now there are two teams play against each other. Their final scores are S and S'. How would you compute the maximum number of times the team that leads could have changed? 
//	 For example, if S=10 and S'=6. The lead could have changed 4 times: 
//	 Team 1 scores 2, then Team 2 scores 3 (lead change); 
//	 Team 1 scores 2 (lead change), Team 2 score 0 (no lead change); 
//	 Team 1 scores 0, Team 2 scores 3 (lead change); 
//	 Team 1 scores 3, Team 2 scores 0 (lead change); 
//	 Team 1 scores 3, Team 2 scores 0 (no lead change).
	 
	 public int maxLeadChange(int[] scores, int s1, int s2){
		 int[][] dp=new int[s1+1][s2+1];
		 for(int i=2;i<=s1;i++)
			 dp[i][0]=1;
		 for(int i=2;i<=s2;i++)
			 dp[0][i]=1;
		 
		 for(int i=1;i<=s1;i++){
			 for(int j=1;j<=s2;j++){
				 for(int k=0;k<scores.length;k++){
					 for(int l=0;l<scores.length;l++){
						 if((i-j)*((i-scores[k])-(j-scores[l]))<0){
							 if(i-scores[k]>=0&&j-scores[l]>=0){
								 dp[i][j]=Math.max(dp[i][j], dp[i-scores[k]][j-scores[l]]+1);
							 }
							 
						 }
						 else {
							 if(i-scores[k]>=0&&j-scores[l]>=0){
								 dp[i][j]=Math.max(dp[i][j], dp[i-scores[k]][j-scores[l]]);
							 }
						 }
					 }
				 }
			 }
		 }
		 
//		 for(int i=0;i<dp.length;i++){
//			 System.out.println(Arrays.toString(dp[i]));
//		 }
		return dp[s1][s2];	 
	 }
	 
	 public int maxLeadChange2(int[] scores, int s1, int s2){
//		 int[] change={0};
		 return maxLeadChangeUtil(scores, s1, s2, 0);
	 }
	 
	 public int maxLeadChangeUtil(int[] scores, int s1, int s2, int c){
		 
		 if(s1==0&&s2==0)
			 return c;
		 else if(s1<0||s2<0)
			 return -1;
		 int res=-1;
		 for(int i=0;i<scores.length;i++){
			 for(int j=0;j<scores.length;j++){
				 if(scores[i]==0&&scores[j]==0)
					 continue;
				 boolean change=(s1-scores[i]-(s2-scores[j]))*(s1-s2)<0;
//				 boolean change=(s1-scores[i]-(s2-scores[j]))<0&&s1-s2>0||(s1-scores[i]-(s2-scores[j]))>0&&s1-s2<0;
				 int subResult= maxLeadChangeUtil(scores, s1-scores[i], s2-scores[j], c+(change?1:0));
//				 System.out.println("sub change is "+subResult);
					if(subResult>res)
						res=subResult;
			 }
		 }
		 return res;
	 }
	 
	 public int solve(int s1, int s2, int[] A, int c)
	    {
	        if (s1 == 0 && s2 == 0) return c;
	        else if (s1 < 0 || s2 < 0) return -1;

	        int max = -1;
	        for (int i = 0; i < A.length; i++)
	        {
	            for (int j = 0; j < A.length; j++)
	            {
	                int sc1 = A[i];
	                int sc2 = A[j];

	                if (sc1 == 0 && sc2 == 0) continue;
	                boolean change = (s1 > s2 && s1 - sc1 < s2 - sc2 ||
	                       (s1 < s2 && s1 - sc1 > s2 - sc2 ));
	                int x = solve(s1-sc1, s2-sc2, A, c + (change ? 1 : 0));
	                if (x > max) max = x;
	            }
	        }

	        return max;
	    }
	 
//	 Write a function in language of your choice that takes in two strings, 
//	 and returns true if they match. 
//	 Constraints are as follows: String 1, the text to match to, will be alphabets and digits. 
//	 String 2, the pattern, will be alphabets, digits, '.' and '*'. 
//	 '.' means either alphabet or digit will be considered as a "match". 
//	 "*" means the previous character is repeat 0 or more # of times.

	 public boolean isMatch(String s, String p) {
		 int len1=s.length();
		 int len2=p.length();
		 if(len2==0)
			 return len1==0;
		 if(len2==1){
			 if(len1==1&&(p.charAt(0)=='.'||p.charAt(0)==s.charAt(0)))
				 return true;
			 else
				 return false;
		 }
		 
		 if(p.charAt(1)!='*'){
			 if(len1>0&&(s.charAt(0)==p.charAt(0)||p.charAt(0)=='.'))
				 return isMatch(s.substring(1),p.substring(1));
			 return false;
		 }
		 else{
			 while(s.length()>0&&(s.charAt(0)==p.charAt(0)||p.charAt(0)=='.')){
				 if(isMatch(s, p.substring(2)))
					 return true;
				 s=s.substring(1);
			 }
			 return isMatch(s,p.substring(2));
		 }
			 
	 }
	 
	 
//		Pattern Matching 
//		---------------- 
//		Characters: a to z 
//		Operators: * + 
//		* -> matches zero or more (of the character that occurs previous to this operator) 
//		+ -> matches one or more (of the character that occurs previous to this operator) 
		
		public boolean isMatching(String s, String p){
//			System.out.println(s+" "+ p);
			if(p.length()==0)
				return s.length()==0;
			if(p.length()==1)
				return s.length()==1&&s.charAt(0)==p.charAt(0);
			
			if(p.charAt(1)!='*'){
				if(p.charAt(1)=='+'){
					while(s.length()>0&&s.charAt(0)==p.charAt(0)){
						if(isMatch(s, p.substring(2)))
							 return true;
						 s=s.substring(1);
					}
					return isMatching(s,p.substring(2));
				}
				else{
					if(s.length()>0&&s.charAt(0)==p.charAt(0))
						return isMatching(s.substring(1),p.substring(1));
					else return false;
				}				
			}
			else{
				while(s.length()>0&&s.charAt(0)==p.charAt(0)){
					if(isMatching(s,p.substring(2)))
						return true;
					s=s.substring(1);
				}
				return isMatching(s,p.substring(2));
			}
		}
	 
	 public boolean isMatch2(String s, String p) {
		 if(s.length()==0)
			 return allstars(p);
		 if(p.length()==0)
			 return s.length()==0;
		 if(p.length()==1){
			 if(s.length()==1&&(s.charAt(0)==p.charAt(0)||p.charAt(0)=='.'))
				 return true;
			 return false;
		 }
		 char p1=p.charAt(0);
		 char p2=p.charAt(1);
		 
		 if(p2!='*'){
			 if(s.length()>0&&s.charAt(0)==p1||p1=='.')
				 return isMatch2(s.substring(1),p.substring(1));
			 return false;
		 }
		 else{
			 if(s.length()>0&&s.charAt(0)==p1||p1=='.')
				 return isMatch2(s.substring(1),p)||isMatch2(s,p.substring(2));
			 else
				 return isMatch2(s, p.substring(2));
		 }
	 }
	 
	 public boolean allstars(String p){
		 if(p.length()%2!=0)
			 return false;
		 for(int i=1;i<p.length();i+=2){
			 if(p.charAt(i)!='*')
				 return false;
		 }
		 return true;
	 }
	 
//	 2D DP，dp[i + 1][j + 1]表示字符串s(0~ i )和p(0~j)的匹配情况。
//
//	 初始状态：dp[0][0] = true;
//
//	 当s[i] == p[j] || p[j] == '.' 则dp[i][j] = dp[i - 1][j - 1]
//
//	 当p[j] == '*'时：分两种情况：
//
//	 1. s[i] != p[j - 2] && p[j - 2] != '.' 则dp[i][j] = dp[i][j - 2];
//
//	 2. else dp[i][j] =  dp[i][j - 2] | dp[i - 1][j];
	 
	 
	 public boolean isMatch3(String s, String p) {
         int height = s.length(),width = p.length();
         boolean[][] dp = new boolean[height + 1][width + 1];
         dp[0][0] = true;
         for(int i = 1; i <= width; i++){
             if(p.charAt(i - 1) == '*') dp[0][i] = dp[0][i - 2];
         }
         for(int i = 1; i <= height; i++){
             for(int j = 1; j <= width; j++){
                 char sChar = s.charAt(i - 1);                
                 char pChar = p.charAt(j - 1);
                 if(sChar == pChar || pChar == '.'){
                     dp[i][j] = dp[i - 1][j - 1];
                 }else if(pChar == '*'){
                     if(sChar != p.charAt(j - 2) && p.charAt(j - 2) != '.'){
                         dp[i][j] = dp[i][j - 2];
                     }else{
                         dp[i][j] =  dp[i][j - 2] | dp[i - 1][j];
                     }
                 }
             }
         }
         return dp[height][width];
     }
	 
	 public String longestPalindrome2(String s) {
		 int n=s.length();
		 boolean[][] dp=new boolean[n][n];
		 int max=1;
		 int start=0;
		 int end=0;
		 for(int i=n-1;i>=0;i--){
			 for(int j=i;j<n;j++){
				 if(i==j||s.charAt(i)==s.charAt(j)&&j-i==1||s.charAt(i)==s.charAt(j)&&dp[i+1][j-1]){
					 dp[i][j]=true;
					 if(j-i+1>max){
						 max=j-i+1;
						 start=i;
						 end=j;
					 }
				 }					 
			 }
		 }
		 return s.substring(start,end+1);
	 }
	 
	 public int lengthOfLongestSubstring(String s) {
		 if(s.length()<2)
			 return s.length();
		 int max=1;
		 int start=0;
		 HashMap<Character, Integer> map=new HashMap<Character, Integer>();
		 for(int i=0;i<s.length();i++){
			 char c=s.charAt(i);
			 if(!map.containsKey(c))
				 map.put(c, i);
			 else{
				 int dup=map.get(c);
				 max=Math.max(max, i-start);
				 for(int j=dup;j>=start;j--)
					 map.remove(s.charAt(j));
				 start=dup+1;
				 map.put(c, i);
			 }
		 }
		 max=Math.max(max, s.length()-start);
		 return max;
	 }
	 
	 public String convertDecToExcel(int num){
		 String s="";
		 while(num>0){
			 num--;
			 char c=(char) ('A'+num%26);
			 num/=26;
			 s=c+s;
		 }
		 return s;
	 }
	 
	 
	 public int convertExcelToDec(String excel){
		 int res=0;
		 for(int i=0;i<excel.length();i++){
			 res=res*26+(excel.charAt(i)-'A'+1);
		 }
		 return res;
	 }
	 
	 
//	 Write a function that takes 2 strings , search returns true if any anagram of string1(needle) is present in string2(haystack)
	// cat, actor -> T
	// car, actor -> F

	public boolean anaStrStr (String needle, String haystack){
		if(haystack.length()<needle.length())
			return false;
		char[] ch=needle.toCharArray();
		Arrays.sort(ch);
		needle=new String(ch);
		for(int i=0;i<haystack.length()-needle.length()+1;i++){
			String str=haystack.substring(i, i+needle.length());
			char[] strChars=str.toCharArray();
			Arrays.sort(strChars);
			String s=new String(strChars);
			if(needle.equals(s))
				return true;
		}
		return false;
	}
	
	public List<List<Integer>> allCombinations(int n){
		List<List<Integer>> res=new ArrayList<List<Integer>>();
		List<Integer> sol=new ArrayList<Integer>();
		allCombinationsUtil(1,n,sol, res, 0);
		return res;
	}
	
	public void allCombinationsUtil(int cur, int n, List<Integer> sol, List<List<Integer>> res, int cursum){
		if(cur==n||cursum>n)
			return;
		if(cursum==n){
			List<Integer> out=new ArrayList<Integer>(sol);
			res.add(out);
		}
		for(int i=cur;i<n;i++){
			cursum+=i;
			sol.add(i);
			allCombinationsUtil(i,n,sol,res,cursum);
			cursum-=i;
			sol.remove(sol.size()-1);
		}
	}
	
	
//	Example : 
//		Given the AP :- 1 3 7 9 11 13 find the missing value "which would be 5 here". 
//
//		Conditions : 
//		Get an user for the length of AP sequence and make sure user provides length is above 3. 
//		Get the input in a single line ex:- "1 3 5 7 9 11" 
//		Provide the solution in O(n) or less if you can.
	
	public int firstMissingInAP(int[] ap){
		int beg=0;
		int end=ap.length-1;
		int diff=Math.min(ap[1]-ap[0],ap[2]-ap[1]);;
		
		while(beg<end){
			int mid=(beg+end)/2;
			int leftDif=ap[mid]-ap[beg];
			int rightDif=ap[end]-ap[mid];
			
			if(leftDif>diff*(mid-beg)){
				if(mid-beg==1)
					return (ap[beg]+ap[mid])/2;
				else
					end=mid;
			}
			else if(rightDif>diff*(end-mid)){
				if(end-mid==1)
					return (ap[mid]+ap[end])/2;
				else
					beg=mid;
			}
			else
				break;
			System.out.println(beg+" "+mid+" "+end);
		}
		return -1;
	}
	
	public  int getMissingTermInAP(int[] arr) {
		int n=arr.length;
		if (n < 3)
			throw new RuntimeException("n cannot be less than 3");
		int total=(n+1)*(arr[0]+arr[n-1])/2;
		int actualSum = 0;
		for (int x : arr)
			actualSum += x;
		return total - actualSum;
	}
	
	public int findMissingInAP(int[] nums){
		int n=nums.length-1;
		int diff=Math.min(nums[1]-nums[0], nums[2]-nums[1]);
		return findMissingInAP(nums,0,n,diff);
	}
	
	public int findMissingInAP(int[] nums, int left, int right, int difference) {
		int middle = (left + right) / 2;
		int predicted = middle * difference + nums[0];
		while (left < right) {
			System.out.println(left+" "+right);
			if (left == right - 1) {
				//the skipped number is between min and max
				return (nums[left] + nums[right])/2;
			}
			else if (nums[middle] > predicted) {
				// the skipped number is on the left
				return  findMissingInAP(nums, left, middle, difference);
			} else{// added?
				// the skipped number is on the right
				return findMissingInAP(nums, middle, right, difference);
			}
		}
		
		return (nums[left] + nums[right-1])/2;
	}
	
	public static int findMissing_binary(int[] array) {
		assert array != null && array.length > 2;
		
		int diff = Math.min(array[2]-array[1], array[1]-array[0]);

		int low = 0, high = array.length-1;
		while(low < high) {
			int mid = (low+high) >>> 1;

			int leftDiff = array[mid]-array[low];
			if(leftDiff > diff * (mid-low)) {
				if(mid-low == 1)
					return (array[mid]+array[low]) / 2;
				
				high = mid;
				continue;
			}
			
			int rightDiff = array[high]-array[mid];
			if(rightDiff > diff * (high-mid)) {
				if(high-mid == 1)
					return (array[high]+array[mid]) / 2;
				
				low = mid;
				continue;
			}
		}
		
		return -1;
	}
	
//	Given a string Sting="ABCSC" Check whether it contains a Substring="ABC"? 
//
//			1)If no , return "-1". 
//			2)If yes , remove the substring from string and return "SC". 
	
	public String RemoveSubstring(String s, String target){ 
		if(s.length()<target.length())
			return "-1";
		int n=target.length();
		String res="";
		for(int i=0;i<s.length();i++){
			if(i+n<=s.length()&&s.substring(i,i+n).equals(target))
				i=i+n-1;
			else
				res+=s.charAt(i);
		}
		return res;
	}
	
	// subarray equals 0
//	1. Given A[i]
//			  A[i] | 2 |  1 | -1 | 0 | 2 | -1 | -1
//			-------+---|----|--------|---|----|---
//			sum[i] | 2 |  3 |  2 | 2 | 4 |  3 |  2
//
//			2. sum[i] = A[0] + A[1] + ...+ A[i]
//			3. build a map<Integer, Set>
//			4. loop through array sum, and lookup map to get the set and generate set, and push <sum[i], i> into map.
//
//			Complexity O(n)
	
	public void subarrayEqualsZero(int[] A){
		HashMap<Integer, List<Integer>> map=new HashMap<Integer, List<Integer>>();
		int sum=0;
		for(int i=0;i<A.length;i++){
			sum+=A[i];
			if(map.containsKey(sum))
				map.get(sum).add(i);
			else{
				List<Integer> set=new ArrayList<Integer>();
				set.add(i);
				map.put(sum, set);
			}				
		}
		
		Iterator<Integer> it=map.keySet().iterator();
		while(it.hasNext()){
			
			int key=it.next();
//			System.out.println("key is "+key+" and set is "+map.get(key));
			int size=map.get(key).size();
			if(key==0){
				for(int i=0;i<size;i++)
					System.out.println("subarray from 0 to "+map.get(key).get(i));
			}
			if(size>1){
				for(int i=0;i<size-1;i++){
					for(int j=i+1;j<size;j++)
						System.out.println("subarray from "+(map.get(key).get(i)+1)+" to "+map.get(key).get(j));
				}
			}
		}
	}
	
	public void continuousSubseq(int[] nums){
		HashMap<Integer,List<Integer>> map=new HashMap<Integer, List<Integer>>();
		int sum=0;
		for(int i=0;i<nums.length;i++){
			sum+=nums[i];
			if(sum==0){
				System.out.println("from 0 to "+ i);
			}
			if(map.containsKey(sum)){
				
				for(int j=0;j<map.get(sum).size();j++){
					System.out.println("from "+ (map.get(sum).get(j)+1)+" to "+i);
				}
				map.get(sum).add(i);
			}
			else{
				List<Integer> pos=new ArrayList<Integer>();
				pos.add(i);
				map.put(sum, pos);
			}
		}
	}
	
	
	
//	给定一个string，只有A，B，C，规定
//	AB->AA, 
//	BA->AA, .鐣欏璁哄潧-涓€浜�-涓夊垎鍦�
//	CB->CC, 
//	BC->CC, 
//	AA->A, 
//	CC->C, 
//	可以使用任意规则化简string，要求return最简的string
//	比如ABBCC->AC
	
	public String simplifyString(String s, HashMap<String, String> rules){
		if(s.length()<2)
			return s;
		int i=0;
		String res="";
		while(i<s.length()-2){
			String sub=s.substring(i,i+2);
			if(rules.containsKey(sub))
				s=res+rules.get(sub)+s.substring(i+2);
			else{
				res+=s.charAt(i);
				i++;
				s=res+s.substring(i);
			}
//			System.out.println(s+" "+i);
		}
		return s;
	}
	
	public int longestPath(TreeNode root){
		if(root==null)
			return 0;
		return Math.max(longestPath(root, true), longestPath(root,false));
	}
	
	public int longestPath(TreeNode root, boolean isLeft){
		if(root==null)
			return 0;
		int left=0;
		int right=0;
		if(root.left!=null&&isLeft)
			left=longestPath(root.left,isLeft);
		else
			left=0;

		if(root.right!=null&&!isLeft)
			right=longestPath(root.right,false);
		else
			right=0;
		return left>right?left+1:right+1;
	}
	
	
	public boolean canStringBeAPalindrome(String s) {
		char[] a=s.toCharArray();
		Arrays.sort(a);
		int oddNum=0;
		int curCharNum=1;
		
		for(int i=1;i<s.length();i++){
			if(a[i]!=a[i-1]){
				if((curCharNum&1)!=0)
					oddNum++;
				curCharNum=1;
			}
			else
				curCharNum++;
		}
		if((curCharNum&1)!=0)
			oddNum++;
		return oddNum<=1;
	}
	
	public boolean canStringBeAPalindrome2(String s) {
		int[] hash=new int[256];
		int oddNum=0;
		for(int i=0;i<s.length();i++)
			hash[s.charAt(i)]++;
		for(int i=0;i<256;i++){
			if(hash[i]%2!=0)
				oddNum++;
		}
		return oddNum<=1;
	}
	
	public boolean isKPalindrome(String s,int k){
		if(s.length()<k+1)
			return true;
		int n=s.length();
		int[][] dp=new int[n+1][n+1];
		for(int i=0;i<=n;i++){
			dp[i][0]=i;
			dp[0][i]=i;
		}
		
		for(int gap=1;gap<=n;gap++){
			for(int i=1;i<=n-gap;i++){
				int j=i+gap;
				if(s.charAt(i-1)==s.charAt(j-1))
					dp[i][j]=dp[i+1][j-1];
				else
					dp[i][j]=Math.min(dp[i+1][j], dp[i][j-1])+1;
			}
		}
		
		for(int i=0;i<=n;i++)
			System.out.println(Arrays.toString(dp[i]));
		System.out.println(dp[1][n]+" "+k);
		return dp[1][n]<=k;
	}
	
	public boolean isKPalindrome2(String s,int k){
		int n=s.length();
		String r=new StringBuilder(s).reverse().toString();
		System.out.println(r);
		int[][] dp=new int[n+1][n+1];
		for(int i=0;i<=n;i++){
			for(int j=0;j<=n;j++)
				dp[i][j]=1000;
		}
		for(int i=0;i<=n;i++){
			dp[i][0]=i;
			dp[0][i]=i;
		}
		
		for(int i=1;i<=n;i++){
			int from=Math.max(1, i-k);
			int to=Math.min(n, i+k);
			for(int j=from;j<=to;j++){
				if(s.charAt(i-1)==r.charAt(j-1)){
					dp[i][j]=dp[i-1][j-1];
				}
				// note that we don't allow letter substitutions
				
				dp[i][j] = Math.min(dp[i][j], 1 + dp[i][j-1]); // delete character j
				dp[i][j] = Math.min(dp[i][j], 1 + dp[i-1][j]); // insert character i
			}
		}
		
		for(int i=0;i<=n;i++)
			System.out.println(Arrays.toString(dp[i]));
		System.out.println(dp[n][n]+" "+k);
		return dp[n][n]<=2*k;
	}
	
	public boolean isKPalindrome3(String s, int k){
		return isKPalindromeUtil(s, 0, s.length()-1)<=k;
	}
	
	public int isKPalindromeUtil(String s, int i, int j){
		if(i>=j)
			return 0;
		if(s.charAt(i)==s.charAt(j))
			return isKPalindromeUtil(s, i+1, j-1);
		else
			return Math.min(isKPalindromeUtil(s,i+1,j), isKPalindromeUtil(s,i,j-1))+1;
	}
	
//	{ "face", "ball", "apple", "art", "ah" }
//	"htarfbp..."
//	根据下面的string去给上面list words排序。
	
	public void reSorting(String[] strs, String rules){
		HashMap<Character, Integer> map=new HashMap<Character, Integer>();
		for(int i=0;i<rules.length();i++)
			map.put(rules.charAt(i), i);
		
		HashMap<String, String> hash=new HashMap<String, String>();
		String[] nums=new String[strs.length];
		for(int i=0;i<strs.length;i++){
			String s=strs[i];
			String num="";
			for(int j=0;j<s.length();j++){
				num+=map.get(s.charAt(j));
			}
			nums[i]=num;
			hash.put(num, s);
		}
		Arrays.sort(nums);
		for(int i=0;i<nums.length;i++){
			strs[i]=hash.get(nums[i]);
		}
	}
	
//	Given an array write a function to print all triplets in the array with sum 0.
//	e.g:
//	Input:
//	Array = [-1, -3, 5, 4]
//	output:
//	-1, -3, 4
	// print the triplets but duplicates them
	
	public void findTriplets(int[] nums){
		HashSet<Integer> set=new HashSet<Integer>();
		for(int i=0;i<nums.length;i++)
			set.add(nums[i]);
		
		for(int i=0;i<nums.length-1;i++){
			for(int j=i+1;j<nums.length;j++){
				int num=0-(nums[i]+nums[j]);
				if(set.contains(num))
					System.out.println(nums[i]+" "+nums[j]+" "+num);
			}
		}
	}
	
//	Problem : Given a sorted set of elements, find the groups of THREE elements which are in Arithmetic Progression.
	/* For  each element (b) except the first and last element,
     * Find a and c (a is in the left of b, c is in the right of b), such that,
     * a+c = 2*b 
     */  
	public List<List<Integer>> findThreeElementsAP(int[] nums){
		List<List<Integer>> res=new ArrayList<List<Integer>>();
		if(nums.length<3)
			return res;
		for(int i=1;i<nums.length-1;i++){
			int j=i-1;
			int k=i+1;
			
			while(j>=0&&k<nums.length){
				int sum=nums[j]+nums[k];
				if(sum==2*nums[i]){
					List<Integer> sol=new ArrayList<Integer>();
					sol.add(nums[j]);
					sol.add(nums[i]);
					sol.add(nums[k]);
					res.add(sol);
					j--;
					k++;
				}
				else if(sum>2*nums[i])
					j--;
				else
					k++;
			}			
		}
		return res;
	}
	
	
//	Let's say there is a double square number X, which can be expressed as the sum of two perfect squares, for example, 10 is double square because 10 = 3^2 + 1^2 
//
//	Determine the number of ways which it can be written as the sum of two squares
	
	public int waysOfPerfectSquare(int x){
		int count=0;
		int i=0;
		int j=(int) Math.sqrt(x);
		while(i<=j){
			if(i*i+j*j==x){
				count++;
				i++;
				j--;
			}
			else if(i*i+j*j>x)
				j--;
			else
				i++;
		}
		return count;
	}
	
	public int FindNearestNum(int[] A, int target){
		int i=0;
		int j=A.length-1;
		int minDif=Integer.MAX_VALUE;
		int res=-100;
		while(i<=j){
			int mid=(i+j)/2;
			int diff=Math.abs(A[mid]-target);
			if(diff<minDif){
				minDif=diff;
				res=A[mid];
			}
			if(A[mid]==target)
				return target;
			else if(A[mid]<target)
				i=mid+1;
			else
				j=mid-1;			
		}
		return res;
	}
	
//	Given an array, find all unique three-member subsets, 
//	with unique being that [0,2,3] and [3,2,0] are the same set. Should run in faster than 2^n time
	
	public List<List<Integer>> findAllTriplets(int[] set){
		List<List<Integer>> res=new ArrayList<List<Integer>>();
		Arrays.sort(set);
		int n=set.length;
		for(int i=0;i<n-2;i++){
			if(i>0&&set[i]==set[i-1])
				continue;
			for(int j=i+1;j<n-1;j++){
				if(j>i+1&&set[j]==set[j-1])
					continue;
				for(int k=j+1;k<n;k++){
					if(k>j+1&&set[k]==set[k-1])
						continue;
					List<Integer> triple=new ArrayList<Integer>();
					triple.add(set[i]);
					triple.add(set[j]);
					triple.add(set[k]);
					res.add(triple);
				}
			}
		}
		return res;
	}
	
	public List<List<Integer>> findAllTriplets2(int[] set){
		List<List<Integer>> res=new ArrayList<List<Integer>>();
		List<Integer> sol=new ArrayList<Integer>();
		Arrays.sort(set);
		boolean[] visited=new boolean[set.length];
		findAllTriplets2(0, set, visited, sol, res, 0);
		return res;
	}
	
	public void findAllTriplets2(int dep, int[] set, boolean[] visited, List<Integer> sol, List<List<Integer>> res, int cur){
		if(dep==3){
			List<Integer> out=new ArrayList<Integer>(sol);
			res.add(out);
		}
		for(int i=cur;i<set.length;i++){
			if(!visited[i]){
				if(i!=0&&set[i]==set[i-1]&&!visited[i-1])
					continue;
				sol.add(set[i]);
				visited[i]=true;
				findAllTriplets2(dep+1,set,visited,sol,res, i+1);
				visited[i]=false;
				sol.remove(sol.size()-1);
			}
		}
	}
	
	
	public static int getLength(ListNode head){
		int len=0;
		ListNode cur=head;
		while(cur!=null){
			cur=cur.next;
			len++;
		}
			return len;
	}
	
	public void reorderListRecur(ListNode head) {
		if(head==null||head.next==null)
			return;
		int n=getLength(head);
		reorderList(head, n);
	}
	
	public ListNode reorderList(ListNode head, int n){
		if(n==0) return null;
        if(n==1) {
            ListNode temp = head.next;
            head.next = null;
            return temp;
        }
        if(n==2) {
            ListNode temp = head.next.next;
            head.next.next = null;
            return temp;
        }
        ListNode node = reorderList(head.next, n-2);
        ListNode tail = node.next;
        node.next = head.next;
        head.next = node;
        return tail;
	}
	
	public static void reorderListRecur2(ListNode head) {
		if(head==null||head.next==null)
			return;
		int len=0;
		ListNode cur=head;
		while(cur!=null){
			len++;
			cur=cur.next;
		}
		reorderListUtil(head, 0, len-1);
	}
	
//	Suppose nodes in the list are indexed from 0 to n-1. 
//	We first make a recursive call to reverse the list with indexes from 1 to n-2. 
//	Also, the recursive call provides us a pointer to the node with index n-2 (this is done via the “pass by reference” i
	public static ListNode reorderListUtil(ListNode head, int left, int right){
		if(left>right){
			return null;
		}
		if(left==right){
			ListNode tail=head;
			return tail;
		}
		if(left==right-1){
			ListNode tail=head.next;
			return tail;
		}
		
		ListNode innerHead=head.next;
		ListNode tail=reorderListUtil(innerHead, left+1, right-1);
		
		ListNode tmp=tail.next;
		tail.next=tmp.next;
		tmp.next=head.next;
		head.next=tmp;
		return tail;
	}
	
	public int longestCommonSubstring(String s1, String s2){
		int m=s1.length();
		int n=s2.length();
		int max=0;
		int[][] dp=new int[m+1][n+1];
		for(int i=1;i<m;i++){
			for(int j=1;j<n;j++){
				if(s1.charAt(i-1)==s2.charAt(j-1)){
					dp[i][j]=dp[i-1][j-1]+1;
					max=dp[i][j]>max?dp[i][j]:max;
				}
				else
					dp[i][j]=0;
			}
		}
		return max;
	}
	
//	FInd the maximum sum of a sub-sequence from an positive integer array 
//	where any two numbers of sub-sequence are not adjacent to each other in the original sequence. 
//	E.g 1 2 3 4 5 6 --> 2 4 6
	public int maxSumNoAdacency(int[]A){
		int n=A.length;
		if(n==0)
			return 0;
		if(n==1)
			return A[0];

		int[] dp=new int[n];
		dp[0]=A[0];
		dp[1]=Math.max(A[0], A[1]);
		
		for(int i=2;i<n;i++){
			dp[i]=Math.max(dp[i-1],dp[i-2]+A[i]);
		}
		return dp[n-1];
	}
	
	public int longestIncreasingSeq2(int[] A){
		int[] dp=new int[A.length];
		for(int i=0;i<A.length;i++)
			dp[i]=1;
		int max=1;
		for(int i=1;i<A.length;i++){
			for(int j=0;j<i;j++){
				if(A[i]>A[j]&&dp[i]<dp[j]+1){
					dp[i]=dp[j]+1;
					max=Math.max(max, dp[i]);
				}
			}
		}
		return max;
	}
	
	//flipdown
	public TreeNode upsideDown(TreeNode root){
		if(root==null||root.left==null&&root.right==null)
			return root;
		TreeNode node=upsideDown(root.left);
		root.left.left=root.right;
		root.left.right=root;
		root.left=root.right=null;
		return node;
	}
	
//	1. binary-search (O(log(n)). If citations[i] >= i then h >= i (if array's in descending order). 
//	2. Here's a O(n) time & space solution in ruby. The trick is you can ignore citation-counts larger than n.

	public int getHindexFromSorted(int[] citation) {
		int low = 0;
		int high = citation.length - 1;
		int idx = (low + high) / 2;
		while (low <= high) {
			if (citation[idx] >= idx + 1) {
				low = idx + 1;
			} else {
				high = idx - 1;
			}
			idx = (low + high) / 2;
		}
		return idx + 1;
	}

	public int getHindexFromUnsorted(int[] citation) {
		int[] hindex = new int[citation.length + 1];
		for (int i = 0; i < citation.length; i++) {
			if (citation[i] >= citation.length) {
				hindex[citation.length]++;
			} else {
				hindex[citation[i]]++;
			}
		}
		int count = 0;
		for (int i = hindex.length - 1; i > 0; i--) {
			if (i >= count) {
				count += hindex[i];
			}
		}
		return count;
	}
	
	
	
	public boolean isMatched(String regex, String str) {
        if (str.length() == 0) {
            // Match is true when regex is exhausted or it's last char is "*" - allowing optional str
            return regex.length() == 0 || regex.charAt(regex.length() - 1) == '*';
        }

        if (regex.length() == 0) {
            // Match is true only if str is fully consumed
            return str.length() == 0;
        }

        Character curReg = regex.charAt(0);
        Character nextReg = regex.length() >= 2 ? regex.charAt(1) : null;
        Character curStr = str.length() != 0 ? str.charAt(0) : null;

        if (nextReg == null || (nextReg != '*' && nextReg != '+')) {
            // This is a simple match - just take the first char from both regex and str and recurse IFF current match is detected
            return isCharMatched(curReg, curStr) && isMatched(regex.substring(1), str.substring(1));
        } else {
            if (nextReg == '*') {
                // The current regex char is followed by "*" - create 2 branches:
                // - one with unmodified regex and reduced str IFF current match detected - meaning to continue repetition if possible
                // - the other one with reduced regex and unmodified str - meaning to try out the optional nature of "*"
                return (isCharMatched(curReg, curStr) && isMatched(regex, str.substring(1)))
                        || isMatched(regex.substring(2), str);
            } else if (nextReg == '+') {
                // The current regex char is followed by "+" - reduce to 1 branch with "*" instead of "+"
                return isCharMatched(curReg, curStr) && isMatched(curReg + "*" + regex.substring(2), str.substring(1));
            } else {
                return false;
            }
        }
    }

    public boolean isCharMatched(Character regexCh, Character strCh) {
        return regexCh == strCh || (regexCh == '.' && strCh >= 'a' && strCh <='z');
    }
    
    
    
//    Given a character array. Find if there exists a path from O to X. Here is an example 
//    . . . . . . . 
//    . . . . . . . 
//    w . . . . . . 
//    w .w.w.. 
//    . . . . O . . 
//    . . w. . . . 
//    . . . X . . . 
//
//    You have to just keep in mind that you cannot go through 'W'.
    
    public boolean findPath(char[][] matrix){
    	boolean[][] visited=new boolean[matrix.length][matrix[0].length];
    	for(int i=0;i<matrix.length;i++){
    		for(int j=0;j<matrix[0].length;j++){
    			if(matrix[i][j]=='O'){
    				if(dfs(matrix, i, j, visited))
    					return true;
    			}
    		}
    	}
    	return false;
    }
    int[] x={0,0,1,1,1,-1,-1,-1};
    int[] y={-1,1,-1,0,1,-1,0,1};
    
    public boolean dfs(char[][] matrix, int i, int j, boolean[][] visited){
    	System.out.println("cur row is "+i+" and cur col is "+j);
    	if(matrix[i][j]=='X')
    		return true;
    	visited[i][j]=true;
    	boolean res=false;
    	for(int k=0;k<8;k++){
    		int newx=i+x[k];
    		int newy=j+y[k];
    		if(newx>=0&&newx<matrix.length&&newy>=0&&newy<matrix[0].length&&!visited[newx][newy]&&matrix[i][j]!='W'){
    			res|=dfs(matrix, newx, newy, visited);
    		}
    	}
    	visited[i][j]=false;
    	return res;
    }
    
    
//    String Reduction 

//    Given a string consisting of a,b and c's, we can perform the following operation: 
//    Take any two adjacent distinct characters and replace it with the third character. 
//    For example, if 'a' and 'c' are adjacent, they can replaced with 'b'.
//    What is the smallest string which can result by applying this operation repeatedly?

    
    public int stringReduction(String s){
    	System.out.println(s);
    	int min=s.length();
    	if(min<2)
    		return s.length();
    	if(min==2)
    		return s.charAt(0)==s.charAt(1)?2:1;
    	for(int i=0;i<s.length()-1;i++){
    		if(s.charAt(i)!=s.charAt(i+1)){
//    			System.out.println("cur is "+s.substring(0,i)+" and "+(s.charAt(i)-'a')+" "+(s.charAt(i+1)-'a')+" and "+s.substring(i+2));
    			s=s.substring(0,i)+(char)((3-(s.charAt(i)-'a')-(s.charAt(i+1)-'a'))+'a')+s.substring(i+2);
    			int temp=stringReduction(s);
    			if(min>temp)
    				min=temp;
    		}
    	}
    	return min;
    }
    
    public int[] multiplication(int[] nums){
    	int[] res=new int[nums.length];
    	res[0]=1;
    	for(int i=1;i<nums.length;i++)
    		res[i]=res[i-1]*nums[i-1];
    	int c=1;
    	for(int i=nums.length-2;i>=0;i--){
    		c*=nums[i+1];
    		res[i]*=c;
    	}
    	return res;
    }
    
    public int[] multiplication2(int[] nums){
    	int n=nums.length;
    	int[] res=new int[n];
    	Arrays.fill(res, 1);
    	
    	int left=1,right=1;
    	for(int i=0;i<n;i++){
    		res[i]*=left;
    		res[n-i-1]*=right;
    		left*=nums[i];
    		right*=nums[n-i-1];
    	}
    	
    	return res;
    }
    
    TreeNode targetLeaf=null;
    List<Integer> res=new ArrayList<Integer>();
//    Find the maximum sum leaf to root path in a Binary Tree
    public int maxRootLeafPathSum(TreeNode root){
    	if(root==null)
    		return 0;
    	int[] maxSum={Integer.MIN_VALUE};
    	getTargetLeaf(root, maxSum, 0);
    	System.out.println("target leaf is "+targetLeaf.val);
    	List<Integer> path=new ArrayList<Integer>();
    	printPath(root, path, targetLeaf);
    	System.out.println(res);
    	return maxSum[0];
    }
    
    public void getTargetLeaf(TreeNode root, int[] maxSum, int cursum){
    	if(root==null)
    		return;
    	cursum+=root.val;
    	if(root.left==null&&root.right==null){
    		if(cursum>maxSum[0]){
    			maxSum[0]=cursum;
    			targetLeaf=root;
    		}
    	}
    	getTargetLeaf(root.left,maxSum,cursum);
    	getTargetLeaf(root.right,maxSum, cursum);
    }
    
    public void printPath(TreeNode root, List<Integer> path, TreeNode target){
    	
    	if(root==null)
    		return;
//    	System.out.println(path+" "+root.val);
    	path.add(root.val);
    	if(root==target){
    		res=new ArrayList<Integer>(path);
    		return;
    	}
    	printPath(root.left,path,target);
    	printPath(root.right,path,target);
    	path.remove(path.size()-1);
//    	if(root==target||printPath(root.left,target)||printPath(root.right,target)){
//    		System.out.print(root.val+" ");
//    		return true;
//    	}
//    	return false;
    }
    
    public int maxSumPath(TreeNode root){
    	int[] max={Integer.MIN_VALUE};
    	maxSumPath(root,max);
    	return max[0];
    }
    
    public int  maxSumPath(TreeNode root, int[] max){
    	if(root==null)
    		return Integer.MIN_VALUE;
    	if(root.left==null&&root.right==null)
    		return root.val;
    	int leftSum=maxSumPath(root.left,max);
    	int rightSum=maxSumPath(root.right,max);
    	
    	//passing through root;
    	if (root.left!=null && root.right!=null)
    		max[0] = Math.max(max[0], root.val + leftSum + rightSum);
    	return Math.max(leftSum, rightSum)+root.val;
    }
    
//    Given a list of words, L, that are all the same length, 
//    and a string, S, find the starting position of the substring of S
//    that is a concatenation of each word in L exactly once and without any intervening characters.
//    This substring will occur exactly once in S.. 
    
    public List<Integer> findSubstring(String S, String[] L) {
    	List<Integer> res=new ArrayList<Integer>();
    	int n=L.length;
    	int len=L[0].length();
    	HashMap<String, Integer> map=new HashMap<String,Integer>();
    	for(String s:L){
    		if(!map.containsKey(s))
    			map.put(s, 1);
    		else
    			map.put(s, map.get(s)+1);
    	}
    	for(int i=0;i<S.length()-n*len+1;i++){
    		HashMap<String,Integer> found=new HashMap<String, Integer>();
    		int j=0;
    		for(;j<n;j++){
    			String sub=S.substring(i+j*len,i+j*len+len);
    			if(!map.containsKey(sub))
    				break;
    			if(found.containsKey(sub))
    				found.put(sub, found.get(sub)+1);
    			else
    				found.put(sub, 1);
    			if(found.get(sub)>map.get(sub))
    				break;
    		}
    		if(j==0)
    			res.add(i);
    	}
    	return res;
    }
//    Given a string, every step you can add/delete/change one character at any position. Minimize the step number to make it a palindrome.
    
    public int minSteps(String s){
    	int n=s.length();
    	int[][] dp=new int[n][n];
    	for(int i=0;i<n-1;i++){
    		if(s.charAt(i)!=s.charAt(i+1))
    			dp[i][i+1]=1;
    	}
    	
    	for(int gap=2;gap<n;gap++){
    		for(int i=0;i<n-gap;i++){
    			int j=i+gap;
    			if(s.charAt(i)==s.charAt(j))
    				dp[i][j]=dp[i+1][j-1];
    			else{
    				dp[i][j]=Math.min(dp[i+1][j-1], Math.min(dp[i+1][j],dp[i][j-1]))+1;
    			}
    		}
    	}
    	return dp[0][n-1];
    }
    
    public double calculateWaterVolum(int c, int l, int kth){ // kth is one-based
    	int[] height=new int[kth];
    	double[] water=new double[kth];
    	int childIndex=0;
    	
    	water[0]=l;
    	
    	for(int i=0;i<kth-1;i++){
    		double over=0.0;
    		
    		if(water[i]>c){
    			over=(water[i]-c)/2.0;
    			water[i]=c;
    		}
    		if(i==0||height[i-1]<height[i])
    			childIndex++;
    		if(childIndex==kth)
    			break;
    		height[childIndex]=height[i]+1;
    		water[childIndex]+=over;
    		childIndex++;
    		
    		if(childIndex==kth)
    			break;
    		height[childIndex]=height[i]+1;
    		water[childIndex]+=over;
    	}
    	return water[kth-1]>c?c:water[kth-1];
    }
    
    
//    Really like the linear solution of this problem. 
//    You have an array of 0s and 1s and you want to output all the intervals (i, j) where the number of 0s and numbers of 1s are equal. 
//
//    Example 
//
//    pos = 0 1 2 3 4 5 6 7 8 
//    0 1 0 0 1 1 1 1 0 
    
    public void intervalWithEqual0sAnd1s(int[] A){
    	for(int i=0;i<A.length;i++){
    		if(A[i]==0)
    			A[i]=-1;
    	}
    	
    	HashMap<Integer, Integer> map=new HashMap<Integer, Integer>();
    	int sum=0;
    	for(int i=0;i<A.length;i++){
    		sum+=A[i];
    		if(sum==0)
    			System.out.println("interval is from 0 to "+i);
    		if(map.containsKey(sum)){
    			System.out.println("interval is from "+ (map.get(sum)+1)+" to "+i);
    			map.put(sum, i);
    		}
    		else
    			map.put(sum, i);
    	}
    }
    
    

    
//    Count words in a sentence. Words can be separated by more than one space.
    public int countWords(String sentence){
    	sentence=sentence.trim();
    	int count=0;
    	char last=sentence.charAt(0);
    	for(int i=1;i<sentence.length();i++){
    		if(sentence.charAt(i)==' '&&last!=' '){
    			count++;
    		}
    		last=sentence.charAt(i);
    	}
    	if(last!=' ')
    		count++;
    	return count;
    }
    
    
    public TreeNode UpsideDownBinaryTree(TreeNode root) {
		TreeNode p=root;
		TreeNode parent=null, parentRight=null;
		while(p!=null){
			TreeNode left=p.left;
			p.left=parentRight;
			parentRight=p.right;
			p.right=parent;			
			
			parent=p;
			p=left;
		}
		return parent;
	}
    
    public TreeNode UpsideDownBinaryTreeWithStack(TreeNode root) {
    	if(root==null)
    		return null;
    	Stack<TreeNode > stk=new Stack<TreeNode>();
    	TreeNode p=root;
    	while(p!=null){
    		stk.push(p);
    		p=p.left;
    	}
    	TreeNode rRoot=stk.pop();
    	TreeNode cur=rRoot;
    	while(!stk.isEmpty()){
    		TreeNode node=stk.pop();
    		TreeNode right=node.right;
    		cur.left=right;
    		cur.right=node;
    		cur=node;
    	}
    	cur.left=cur.right=null;
    	return rRoot;
    }
    
//    finding the maximum tree length----diameter
    
    public int maxTreeLength(TreeNode root){
    	if(root==null)
    		return 0;
    	int left=maxTreeLength(root.left);
    	int right=maxTreeLength(root.right);
    	int leftH=getHeight(root.left);
    	int rightH=getHeight(root.right);
    	
    	int arch=leftH+rightH+1;
    	return Math.max(left, right)>arch?arch:Math.max(left, right);
    }
    
    public int maxTreeLength2(TreeNode root){
    	int[] h={0};
    	return maxTreeLength2(root, h);
    }
    
    public int maxTreeLength2(TreeNode root, int[] h){
    	
    	if(root==null){
    		h[0]=0;
    		return 0;
    	}
    	int[] lh={0};
    	int[] rh={0};
    	
    	int leftD=maxTreeLength2(root.left,lh);
    	int rightD=maxTreeLength2(root.right,rh);
    	h[0]=Math.max(lh[0], rh[0])+1;
//    	System.out.println("height is "+h[0]);
    	return Math.max(Math.max(leftD, rightD),lh[0]+rh[0]+1);
    	
    }
    
    
    public int numOfPalindromes(String s){
    	int n=s.length();
    	int num=0;
    	boolean[][] dp=new boolean[n][n];
    	for(int i=0;i<n;i++){
    		dp[i][i]=true;
    		num++;
    	}
    	
    	for(int gap=1;gap<n;gap++){
    		for(int i=0;i<n-gap;i++){
    			int j=i+gap;
    			
    			if(i+1==j&&s.charAt(i)==s.charAt(j)){
    				dp[i][j]=true;
    				num++;
    			}
    			else if(i+1<j){
    				if(s.charAt(i)==s.charAt(j)&&dp[i+1][j-1]){
    					dp[i][j]=true;
    					num++;
    				}    					
    			}
    		}
    	}
    	return num;	
    }
    
    
//    Given a unordered array of numbers, remove the fewest number of numbers to produce the longest ordered sequence. Print count of fewest numbers to be removed, and the remaining sequence. 
//    For example, if input is 1 2 3 4 5 6 7 8 9 10, no (zero) numbers are removed, and input is the longest sequence. 
//    If input is, 1 2 3 8 10 5 6 7 12 9 4 0, then fewest number of elements to be removed is 5 is the fewest number to be removed, and the longest sequence is 1 2 3 5 6 7 12.
    
    
    public int LeastRemoval(int[] A){
    	int max=1;
    	int n=A.length;
    	int[] dp=new int[n];
    	String[] path=new String[A.length];
    	for(int i=0;i<dp.length;i++){
    		dp[i]=1;
    		path[i]=A[i]+" ";
    	}
    	for(int i=1;i<n;i++){
    		for(int j=0;j<i;j++){
    			if(A[i]>A[j]&&dp[i]<dp[j]+1){
    				dp[i]=dp[j]+1;
    				path[i]=path[j]+A[i]+" ";
    				max=Math.max(max, dp[i]);
    			}
    		}
    	}
    	for(int i=0;i<A.length;i++){
    		if(dp[i]==max)
    			System.out.println(path[i]);
    	}
    	System.out.println();
    	return A.length-max;
    }
    // if nonnegative numbers 
    public  boolean subSequenceSumToTotal(int[] A, int total){
    	if(A.length==0)
    		return false;
    	int sum=A[0];
    	int j=0;
    	for(int i=1;i<A.length;i++){
    		while(sum>total&&j<i-1){
    			sum-=A[j++];
    		}
    		if(sum==total)
    			return true;
    		sum+=A[i];
    		if(sum==total)
    			return true;
    		System.out.println("i="+i+" and sum="+sum);
    	}
    	return false;
    }
    // with negative numbers 
    public boolean subSequenceSumToTotal2(int[] A, int total){
    	HashMap<Integer, Integer> map=new HashMap<Integer, Integer>();
    	int sum=0;
    	map.put(0, -1);
    	for(int i=0;i<A.length;i++){
    		sum+=A[i];
    		if(map.containsKey(sum-total)){
    			System.out.println("from "+(map.get(sum-total)+1)+" to "+i);
    			return true;
    		}
    		else
    			map.put(sum,i);
    	}
    	return false;
    }
    
    
    
    
//    Given a histogram of n items stacked next to each other, find the Max area under a given rectangle. 
//    Each bar in the histogram has width = 1 unit and hight is variable. 
    public int largestRectangleArea(int[] height) {
		int[] h=Arrays.copyOf(height, height.length+1); // last dummy 0 is to pop up all the values in stack
		Stack<Integer> stk=new Stack<Integer>();
		int max=0;
		for(int i=0;i<h.length;){
			if(stk.isEmpty()||h[i]>=h[stk.peek()])
				stk.push(i);
			else{
				int top=stk.pop();
				int length=stk.isEmpty()?i:i-stk.peek()-1;
				max=Math.max(max, h[top]*length);
			}
		}
		return max;
	}
    
//    Find and delete nodes from a linked list with value=k. What's the complexity? Does it handle boundary cases? 
    
    public ListNode removeNodes(ListNode head, int k){
    	ListNode dummy=new ListNode(0);
    	dummy.next=head;
    	ListNode cur=head;
    	ListNode pre=dummy;
    	while(cur!=null){
    		if(cur.val==k){
    			pre.next=cur.next;
    			cur=cur.next;
    		}
    		else{
    			pre=cur;
    			cur=cur.next;
    		}
    	}
    	return dummy.next;
    }
    
    public int minDepth(TreeNode root) {
    	if(root==null)
    		return 0;
    	int left=minDepth(root.left);
    	int right=minDepth(root.right);
    	if(left==0)
    		return right+1;
    	if(right==0)
    		return left+1;
    	return Math.min(left, right)+1;
    }
    
    public int minDepthBFS(TreeNode root) {
    	if(root==null)
    		return 0;
    	Queue<TreeNode> que=new LinkedList<TreeNode>();
    	int curLevel=0;
    	int nextLevel=0;
    	que.offer(root);
    	curLevel++;
    	int dep=1;
    	while(!que.isEmpty()){
    		TreeNode top=que.poll();
    		curLevel--;
    		if(top.left==null&&top.right==null)
    			return dep;
    		if(top.left!=null){
    			que.add(top.left);
    			nextLevel++;
    		}
    		if(top.right!=null){
    			que.add(top.right);
    			nextLevel++;
    		}
    		if(curLevel==0){
    			curLevel=nextLevel;
    			nextLevel=0;
    			dep++;
    		}
    	}
    	return dep;
    }
    
    
    public int findKthSmallest(int[] A, int k){
    	int l=0;
    	int r=A.length-1;
    	return findKthSmallest(A, l,r,k);
    }
    
    public int findKthSmallest(int[] A, int l, int r, int k){
    	if(k>0&&k<=r-l+1){
//    		int pivot=partition(A, l, r);
    		int pivot=randomPartition(A, l, r);
    		if(pivot-l==k-1)
    			return A[pivot];
    		else if(pivot-l<k-1)
    			return findKthSmallest(A,pivot+1,r,k-(pivot-l+1));
    		else
    			return findKthSmallest(A, l, pivot-1,k);
    	}
    	return Integer.MAX_VALUE;
    }
    
    public int partition(int[] A, int l, int r){
    	int pivot=A[r]; 
    	int i=l;
    	
    	for(int j=l;j<=r-1;j++){
    		if(A[j]<=pivot){
    			swap(A,i,j);
    			i++;
    		}
    	}
    	swap(A,i,r);
    	return i;
    }
    public void swap(int[] A, int i, int j){
    	int t=A[i];
    	A[i]=A[j];
    	A[j]=t;
    }
    
    public int randomPartition(int[] A, int l, int r){
    	int n=r-l+1;
    	int pivot=(int) (Math.random()*n);
    	swap(A, l+pivot, r);
    	
    	return partition(A, l, r);
    }
    
    public int longestZigZag(int[] sequence){
    	int n=sequence.length;
    	int[] dp=new int[n];
    	for(int i=0;i<n;i++)
    		dp[i]=1;
    	int[] signs=new int[n];
    	
    	int max=1;
    	for(int i=1;i<n;i++){
    		for(int j=0;j<i;j++){
    			if(j==0){
    				signs[i]=sequence[i]-sequence[j];
    				dp[i]=signs[i]==0?dp[i]:dp[j]+1;
    			}
    			if(signs[j]<0&&(sequence[i]-sequence[j])>0||signs[j]>0&&sequence[i]-sequence[j]<0&&dp[i]<dp[j]+1){
    				dp[i]=dp[j]+1;
    				max=Math.max(max, dp[i]);
    				signs[i]=sequence[i]-sequence[j];
    			}
    		}
    	}
//    	System.out.println(Arrays.toString(dp));
    	return dp[n-1];
    }
    
    public int minChange(int[] coins, int m){
    	int n=coins.length;
    	int[] dp=new int[m+1];
    	for(int i=1;i<=m;i++)
    		dp[i]=Integer.MAX_VALUE;
    	
    	for(int i=0;i<=m;i++){
    		for(int j=0;j<n;j++){
    			if(coins[j]<=i&&dp[i-coins[j]]+1<dp[i])
    				dp[i]=dp[i-coins[j]]+1;
    		}
    	}
    	
    	return dp[m];
    }
    
    public int badNeighbors(int[] donations){
    	int n=donations.length;
    	if(n==0)
    		return 0;
    	if(n==1)
    		return donations[1];
    	if(n==2)
    		return Math.max(donations[0], donations[1]);
    	int[] dp=new int[n];
    	
    	int res1=0;
    	dp[0]=donations[0];
    	
    	for(int i=2;i<n-1;i++){
    		dp[i]=Math.max(dp[i-2]+donations[i], dp[i-1]);
    	}
    	res1=dp[n-2];
    	for(int i=0;i<n;i++)
    		dp[i]=0;
//    	dp[0]=donations[0];
    	dp[1]=donations[1];
    	
    	for(int i=2;i<n;i++){
    		dp[i]=Math.max(dp[i-2]+donations[i], dp[i-1]);
    	}
    	return res1>dp[n-1]?res1:dp[n-1];
    }
    
    public int getNthUglyNum(int n){
    	int[] ugly=new int[n];
    	int next_multiple_of_2 = 2;
    	int next_multiple_of_3 = 3;
    	int next_multiple_of_5 = 5;
    	int i2=0, i3=0, i5=0;
    	
    	ugly[0]=1;
    	
    	for(int i=1;i<n;i++){
    		int nextNum=Math.min(next_multiple_of_2, Math.min(next_multiple_of_3, next_multiple_of_5));
    		ugly[i]=nextNum;
    		if(nextNum==next_multiple_of_2){
    			next_multiple_of_2 = ugly[++i2]*2;
    		}
    		if(nextNum==next_multiple_of_3){
    			next_multiple_of_3=ugly[++i3]*3;
    		}
    		if(nextNum==next_multiple_of_5){
    			next_multiple_of_5=ugly[++i5]*5;
    		}
    	}
    	return ugly[n-1];
    }
    
    
    
//    two sum (did before) -> similar to sort color
//
//    boolean is_low(int a)
//    boolean is_med(int a)
//    boolean is_high(int a)
//   
//    input: [-9 (low), 10(high), 4 (med), 7(low), 3(high), 50(med)]
//    output: [-9, 7, 4, 50, 10, 3]
//
//    follow up:. 
//    int rank(int a) -> 0 to k - 1. 
//
//    k ranks, then how to sort the array 
//    1, TreeMap stores <rank, List<Integer>>
//    2, No extra space: similar to insertion sort

    public boolean is_low(int a){
    	return a<0;
    }
    
    public boolean is_med(int a){
    	return a>=0&&a<10;
    }
    
    public boolean is_high(int a){
    	return a>=10;
    }
    
    public void sortLowMedHight(int[] A){
    	if(A.length<2)
    		return;
    	int i=0, j=A.length-1, k=A.length-1;
    	
    	while(i<=j){
    		int a=A[i];
    		if(is_high(a)){
    			swap(A, i, k--);
    			if(j>k)
    				j=k;
    		}
    		else if(is_med(a)){
    			swap(A, i, j--);
    		}
    		else
    			i++;
    	}
    }
    
    public int rank(int a){
    	return a;
    }
    
    public void sortKRanks(int[] A){
    	if(A.length<2)
    		return;
    	
    	for(int i=0;i<A.length;i++){
    		int a=rank(A[i]);
    		for(int j=0;j<i;j++){
    			int b=rank(A[j]);
    			if(a<=b)
    				swap(A,i,j);
    		}
    	}
    }
    
//    find the maximum number in an integer array. The numbers in the array increase first, then decreases. Maybe there’s only increase or decrease.
    public int findMax(int[] A){
    	return findMax(A, 0, A.length-1);
    }
    
    public int findMax(int[] A, int i, int j){
    	if(i==j)
    		return A[i];
    	if(i+1==j)
    		return A[i]>A[j]?A[i]:A[j];
    	int mid=(i+j)/2;
    	if(A[mid]>A[mid-1]&&A[mid]>A[mid+1])
    		return A[mid];
    	else if(A[mid]>A[mid-1]&&A[mid]<A[mid+1])
    		return findMax(A,mid+1,j);
    	return findMax(A, i,mid-1);
    	
    }
    
    public int findMin(int[] num){
    	int left=0;
    	int right=num.length-1;
    	if(num[left]<num[right])
    		return num[left];
    	while(left<right){
    		int mid=(left+right)/2;
    		if(num[mid]>num[right])
    			left=mid+1;
    		else
    			right=mid;
    	}
    	return num[left];
    }
    
    public boolean canJump(int[] A) {
    	boolean[] dp=new boolean[A.length];
    	dp[A.length-1]=true;
    	int gap=0;
    	for(int i=A.length-2;i>=0;i--){
    		if(A[i]>gap){
    			dp[i]=true;
    			gap=0;
    		}
    		else
    			gap++;
    	}
    	return dp[0];
    }
    
    public boolean canJumpII(int[] A) {
    	int m=0;
    	for(int i=0;i<A.length;i++){
    		if(i<=m){
    			m=Math.max(m, A[i]+i);
    			if(m>=A.length-1)
    				return true;
    		}
    	}
    	return false;
    }
//    public int findMaxSubLists(ListNode head1, ListNode head2){
//    	if(head1==null&&head2==null)
//    		return 0;
//    	if(head1==null||head2==null)
//    		return head1==null?head2.val+findMaxSubLists(head1, head2.next):head1.val+findMaxSubLists(head1.next, head2);
//    	if(head1.val>head2.val)
//    		return head1.val+findMaxSubLists(head1.next,head2);
//    	else if(head1.val<head2.val)
//    		return head2.val+findMaxSubLists(head1, head2.next);
//    	else
//    		return head1.val+findMaxSubLists(head1.next,head2.next);
//    }
	
//	 Write all solutions for a^3+b^3 = c^3 + d^3, where a, b, c, d lie between [0, 10^5]
		
//	Design a data structure that supports insert, delete min, delete max, get min, and get max, all in log(n) time.  
	
	
//  Clone a connected undirected graph. Input is a node*. Return the node* of the cloned graph. 
//
//  struct node 
//  { 
//  int value; 
//  vector<Node*> neighbors; 
//  }
  
  

		
	 
	 


	 
//	 word break
	 
//	Convert a binary tree into a circular doubly linked list.  --inorder
	
	
//	Gray code

	
//	 closest path to various sets of nodes, along with being able to detect levels of all nodes 
	

	

	
//	Give a set of objects and a function. Pass two objects to that function and
//	it can tell you whether one object points to another one. 
//	Find one object that is pointed by all other objects.
	
	
	
//	You are given 2 streams of data, representing very sparse vectors
//	you are guaranteed that the 2 incoming streams are of same size
//	give a data structure which is optimized for producing the dot product of those sparse vectors
//	analyze your runtime/space complexity,
//	b) what if you are now told that v1, is much more sparse than v2
//	give another (or the same) data structure optimized for the dot product of any such 2 vectors (where 1 is more sparse than the other)
//	analyze your runtime/space complexity,
	
	
	

	

	
//	1. Modified permutation
//	2. Design news feed
//	4. Sort graph points  
	


//atof", which means convert a string float (e.g. "345.44E-10") to an actual float without using any existing Parse Float functions. 
	 
	 public double atof(String s) {
		 s=s.trim();
		 if(s.length()==0)
			 return 0.0;
		 double integerPart=0;
		 double fractionPart=0;
		 int fractionDivisor=1;
		 boolean isFractional=false;
		 int sign=1;
		 
		 int i=0;
		 if(s.charAt(i)=='-'){
			 sign=-1;
			 i++;
		 }
		 else if(s.charAt(i)=='+')
			 i++;
		 while(i<s.length()){
			 char c=s.charAt(i);
			 if(c>='0'&&c<='9'){
				 if(isFractional){
					 fractionPart=fractionPart*10+(c-'0');
					 fractionDivisor*=10;
				 }
				 else
					 integerPart=integerPart*10+(c-'0');
			 }
			 else if(c=='.'){
				 if(isFractional){
					 return sign*(integerPart+fractionPart/fractionDivisor);
				 }
				 else
					 isFractional=true;
			 }
			 i++;
		 }
		 return sign*(integerPart+fractionPart/fractionDivisor);
	 }
	 public double atof2(String s){
			if(s.length() == 0) return 0.0;
			s=s.trim();
			
			int sign = s.charAt(0) == '-' ? -1 : 1;
			
			if(s.charAt(0) == '-' || s.charAt(0) == '+') s = s.substring(1);
			
			double res = 0.0;
			String intP = "", fracP = "", eP = "";
			
			int eIndex=s.indexOf('E');
			if(eIndex!=-1) {
				eP = s.substring(eIndex+1);
				s = s.substring(0, eIndex);
			}
			
			int dotIndex=s.indexOf('.');
			if(dotIndex==-1)
				intP = s;
			else {
				intP = s.substring(0, dotIndex);
				fracP = s.substring(dotIndex+1);
			}
			
			for(int i=0;i<intP.length();i++)
				res=res*10+(intP.charAt(i)-'c');
			
			for(int i = 0; i < fracP.length(); i++)
				res += pow(10, (i+1)*-1)*(fracP.charAt(i)-'0');
			
			if(eP.length() != 0) {
				int e=0;
				for(int i=0;i<eP.length();i++){
					e=e*10+(eP.charAt(i)-'0');
					res*=Math.pow(10, e);
				}
			}
			
			return res*sign;
		}
		
		
	 public int longestValidParentheses(String s) {
		 int n=s.length();
	        if(n<2)
	            return 0;
	        Stack<Integer> stk=new Stack<Integer>();
	        int start=0;
	        int max=0;
	        for(int i=0;i<n;i++){
	            if(s.charAt(i)=='(')
	                stk.push(i);
	            else{
	                if(stk.isEmpty())
	                    start=i+1;
	                else{
	                    stk.pop();
	                    if(stk.isEmpty())
	                        max=Math.max(max, i-start+1);
	                    else
	                        max=Math.max(max, i-stk.peek());
	                }
	            }
	        }
	        return max;
	 }
	 
	 public boolean validParenthesis(String s){
		 if(s.length()<2)
			 return false;
		 int top=0;
		 for(int i=0;i<s.length();i++){
			 if(s.charAt(i)=='(')
				 top++;
			 else{
				 top--;
			 }
		 }
		 return top==0;
	 }
	 
	 public int divide2(int dividend, int divisor) {
	        boolean neg=(dividend<0&&divisor>0)||(dividend>0&&divisor<0);
	        boolean overflow=false;
	        
	        long dvd=Math.abs((long)dividend);
		    long dvs=Math.abs((long)divisor);
		        
		    int res=0;
		    while(dvd>=dvs){
		        int shift=0;
		        while(dvd>=(dvs<<shift)){
		            shift++;
		        }
		        dvd-=dvs<<(shift-1);
		        if(Integer.MAX_VALUE-(1<<(shift-1))<res){
		            overflow=true;
		            break;
		        }
		        res+=1<<(shift-1);
		    }
		    if(neg){
		        if(overflow)
		            return Integer.MIN_VALUE;
		        return -res;
		    }
		   else{
		       if(overflow)
		            return Integer.MAX_VALUE;
		       return res;
		   }
	    }
	 
	 public int removeDuplicates2(int[] A) {
	        if(A.length<3)
	            return A.length;
	        int j=1;
	        int count=1;
	        for(int i=1;i<A.length;i++){
	            if(A[i]==A[i-1])
	                count++;
	            else
	                count=1;
	            if(count<=2)
	                A[j++]=A[i];
	        }
	        return j;
	    }
	 
//	 [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]这样的数组，要求找到第一个1的位置。
	 
	 public int findFirstOne(int[] A){
		 int i=0;
		 int j=A.length-1;
		 while(i<j){
			 int mid=(i+j)/2;
			 if(mid>0&&A[mid]>A[mid-1])
				 return mid;
			 else if(mid<A.length-1&&A[mid]<A[mid+1])
				 return mid+1;
			 else if(A[mid]==0)
				 i=mid+1;
			 else
				 j=mid-1;
		 }
		 return -1;
	 }
	 
	 public int maximalRectangle(char[][] matrix) {
		 int m=matrix.length;
		 if(m==0)
			 return 0;
		 int n=matrix[0].length;
		 int[][] dp=new int[m][n];
		 
		 for(int i=0;i<n;i++){
			 dp[0][i]=matrix[0][i]=='0'?0:1;
		 }
		 
		 for(int i=1;i<m;i++){
			 for(int j=0;j<n;j++){
				 dp[i][j]=matrix[i][j]=='0'?0:dp[i-1][j]+1;
			 }
		 }
		 
		 int max=0;
		 for(int i=0;i<m;i++){
			 for(int j=0;j<n;j++){
				 int maxLen=dp[i][j];
				 for(int k=j;k>=0;k--){
					 if(dp[i][k]==0)
						 break;
					 maxLen=Math.min(dp[i][k], maxLen);
					 max=Math.max(max, maxLen*(j-k+1));
				 }
				
			 }
		 }
		 return max;
	 }
	 
	 
	 public int numDecodings(String s) {
		 int n=s.length();
		 if(s.length()==0)
			 return 0;
		 int[] dp=new int[n+1];
		 dp[0]=1;
		 if(s.charAt(0)!='0')
			 dp[1]=1;
		 
		 for(int i=2;i<=n;i++){
			 char c1=s.charAt(i-1);
			 if(c1>='1'&&c1<='9')
				 dp[i]=dp[i-1];
			 char c2=s.charAt(i-2);
			 if(c2=='1'||c2=='2'&&c1<='6')
				 dp[i]+=dp[i-2];
		 }
		 return dp[n];
	 }
	 
	 public List<String> decoding(String s){
		 List<String> res=new ArrayList<String>();
		 if(s.length()==0||s.charAt(0)=='0')
			 return res;
		 decoding(s, "", res);
		 return res;
	 }
	 
	 public void decoding( String s, String sol, List<String> res){
		 if(s.length()==0){
			 res.add(sol);
			 return;
		 }
		 
		 for(int i=0;i<2&&i<s.length();i++){
			 if(isValidNum(s.substring(0,i+1))){
				 decoding(s.substring(i+1), sol+(char)(Integer.parseInt(s.substring(0,i+1))+64), res);
			 }
		 }
			 
	 }
	 
	 public boolean isValidNumber(String s) {
			if (s.charAt(0) == '0')
				return false;
			int num = Integer.parseInt(s);
			return num >= 1 && num <= 26;
		}

	public static void main(String[] args){
		Solution sol=new Solution();
		System.out.println(sol.letterCombinations("23"));
		
		ListNode head=new ListNode(1);
		ListNode node1=new ListNode(2);
		ListNode node2=new ListNode(3);
		ListNode node3=new ListNode(4);
		ListNode node4=new ListNode(5);
		ListNode node5=new ListNode(6);
		ListNode node6=new ListNode(7);
		head.next=node1;
		node1.next=node2;
		node2.next=node3;
		node3.next=node4;
		node4.next=node5;
		node5.next=node6;
		
		
		reorderListRecur2(head);
		ListNode cur=head;
		while(cur!=null){
			System.out.print(cur.val+" ");
			cur=cur.next;
		}
		System.out.println();
//		node4.next=head;
//		ListNode inserted=sol.insert(head, 6);
//		ListNode cur=inserted;
//		do{
//			System.out.println(cur.val+" ");
//			cur=cur.next;
//		}while(cur!=inserted);
//		ListNode res=sol.reverseList(head);
		
		ListNode res=sol.reverseListRecur(head);
		while(res!=null){
			System.out.print(res.val+" ");
			res=res.next;
		}
		System.out.println();
		char[] A={'A', 'B', 'C', 'D', 'E', 'F'};
		int[] index={2, 5, 4, 3, 0, 1};
		sol.reArrange(A,index);
		
		TreeNode root = new TreeNode(5);
		root.left = new TreeNode(2);
		root.left.right = new TreeNode(4);
//		root.left.right.left = new TreeNode(3);
		root.left.left = new TreeNode(1);

		root.right = new TreeNode(8);
		root.right.left = new TreeNode(6);
//		root.right.left.right = new TreeNode(7);
		root.right.right = new TreeNode(11);
		root.right.right.left = new TreeNode(9);
		root.right.right.left.right = new TreeNode(13);
		
		System.out.println(sol.longestPath(root));
		System.out.println("ooooooooooooooo");
		System.out.println(sol.printSpecificLevelOrder(root));
		System.out.println("ooooooooooooooo");
		System.out.println(sol.getLeaves(root));
		System.out.println("ooooooooooooooo");
		sol.levelAverage(root);
		System.out.println("---------------------");
		sol.levelAverage2(root);
		System.out.println("ooooooooooooooo");
		System.out.println("----------------");
		System.out.println(sol.pathToLeaf(root));
		
		System.out.println("----------------");
		sol.printAllRootToLeafPaths(root, new ArrayList<Integer>());
		
		System.out.println("***************");
		System.out.println(sol.kthSmallestNode(root,9));
		System.out.println("***************");
		sol.printTreeVertical(root);
		
		System.out.println(sol.lowestCommonAncestor(root, root.right, root.left).val);
		System.out.println(sol.sqrt(9));
		System.out.println(sol.decodeWays("12"));
		
		int arr1[] = {11, 1, 4,13, 21, 3, 7};
	    int arr2[] = {11, 3, 7, 5,1};
	    System.out.println(sol.isSubset3(arr1,arr2));
	    System.out.println(sol.isSubset4(arr1,arr2));
	    
	    System.out.println(sol.addBinary("11", "11"));
		
	    
	    TreeNode r=new TreeNode(0);
	    r.left=new TreeNode(1);
	    r.right=new TreeNode(0);
	    r.left.right=new TreeNode(3);
	    r.right.left=new TreeNode(2);
	    r.right.left.right=new TreeNode(4);
	    
	    System.out.println(sol.pathToLeaf(r));
	    
	    System.out.println("^^^^^^^^^^^^^^^^^^^^^^^^");
	    sol.inorder(r);
	    System.out.println();
	    sol.sinkZeroNode(r);
	    sol.inorder(r);
	    System.out.println();
	    
	    System.out.println(sol.anagram("132", "312"));
	    System.out.println(sol.anagram2("132", "312"));
	    int[] arr= { 10, 22, 9, 33, 21, 50, 41, 60, 80 };
	    System.out.println("longest increasing sequence length is "+ sol.longestIncreasingSeq(arr));
	    System.out.println(sol.longestIncreasingSeq2(arr));
	    System.out.println();
	    System.out.println(sol.fibonacciRecur(10)+" "+sol.fibonacci(10));
//	    System.out.println(sol.fibonacci(10));
	    
	    ListNode l1=new ListNode(1);
	    l1.next=new ListNode(3);
	    ListNode l2=new ListNode(2);
	    l2.next=new ListNode(4);
	    
	    ListNode mh=sol.mergeTwoLists2(l1, l2);
	    while(mh!=null){
	    	System.out.print(mh.val+" ");
	    	mh=mh.next;
	    }
	    
	    System.out.println();
	    System.out.println(sol.countAndSay(4));
	    int[] num={1,3,4,2};
	    sol.getNextSmaller(num);
	    System.out.println(Arrays.toString(num));
	    
	    System.out.println(sol.simplifyPath("/a/./b/../../c/"));
	    System.out.println(sol.multiply("123", "45"));
	    System.out.println(sol.panlindromes("I, am A man!!!","Nam a MA?! I?!"));
	    
	    Point p1=new Point(1,1);
	    Point p2=new Point(1,2);
	    Point p3=new Point(-1,1);
	    Point p4=new Point(1,-1);
	    Point p5=new Point(0.5,1);
	    Point p6=new Point(1.3,1);
	    Point p7=new Point(1,1.5);
	    Point p8=new Point(-1,-.05);
	    Point p9=new Point(-2,3);
	    Point p10=new Point(-0.1,0.5);
	    Point p11=new Point(3,1);
	    
	    Point[] ps={p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11};
	    System.out.println(sol.getKNearestPoints(ps, 4));
	    
//	    [-10, 10], [50, 100], [0, 20] [-5,15]
	    int[] arrs={-10,50,0,-5};
	    int[] deps={10,100,20,15};
	    Interval i1=new Interval(-10,10);
	    Interval i2=new Interval(50,100);
	    Interval i3=new Interval(0,20);
	    Interval target=new Interval(-5,15);
	    List<Interval> intervals=new ArrayList<Interval>();
	    intervals.add(i1);
	    intervals.add(i2);
	    intervals.add(i3);
	    
	    System.out.println(sol.covered(intervals, target));
	    
	    
	    intervals.add(target);
	    
	    System.out.println(sol.maxSubsetNoOverlapping(intervals));
//	    List<Interval> interval_res=sol.maxSubsetNoOverlapping(intervals);
//	    for(int i=0;i<interval_res.size();i++){
//	    	System.out.print(interval_res.get(i)+" ");
//	    }
//	    System.out.println();
	    System.out.println("max overlapping intervals is "+sol.maxIntervalOverlapping(intervals));
	    System.out.println("max overlapping intervals is "+sol.findPlatform(arrs, deps));
	    System.out.println(sol.longestPalindrome("bb"));
	    System.out.println(sol.addNumbers("189","92"));
	    
	    System.out.println(sol.numDecodingsDP("120")+" "+sol.numDecodingsDP2("120"));
	    
	    System.out.println(sol.generateParenthesis(3));
	    
	    int[][] bipartite={{0, 1, 0, 1},
	            {1, 0, 1, 0},
	            {0, 1, 0, 1},
	            {1, 0, 1, 0}
	        };
	    
	    System.out.println(Arrays.toString(sol.separateBipartite(bipartite)));
	    System.out.println(sol.stringPermutation("abc"));
	    System.out.println(sol.stringPermutation2("abc"));
	    
	    List<Integer>  nums=new ArrayList<Integer>();
	    nums.add(3);
	    nums.add(4);
	    nums.add(5);
	    System.out.println(sol.hammingDist(nums));
	    
	    int[] bubbles={84,	69,	76,	86,	94,	91};
	    sol.bubbleSort(bubbles);
	    System.out.println(Arrays.toString(bubbles));
	    
	    int[] coins={1,2,3};
	    System.out.println(sol.coinChange(coins, 4)+", "+sol.coinChange2(coins, 4));
	    
	    double y=29;
	    double start=5.0;
	    double end=7.0;
	    System.out.println(sol.invert(y, start, end));
	    
	    System.out.println(sol.pow(3.2, 5)+", "+sol.iter_power(3.2, 5));
	    
	    System.out.println(sol.longestCommonSubsequence("", ""));
	    System.out.println(sol.sqrt(5.0));
	    
	    int[] house={6,1,2,7,1};
	    System.out.println(sol.maximumSum(house));
	    
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
	    ComplexNode resCNode=sol.flattenList2(chead);
//	    10 5 12 7 11 4 20 13 17 6 2 16 9 8 3 19 15 
	    while(resCNode!=null){
	    	System.out.print(resCNode.val+" ");
	    	resCNode=resCNode.next;
	    }
	    System.out.println();
	    String[] strs={"tsar", "rat", "tar", "star", "tars", "cheese"};
	    System.out.println(sol.anagrams(strs));
	    
	    ListNode h=new ListNode(0);
	    h.next=new ListNode(1);
	    h.next.next=new ListNode(2);
	    h.next.next.next=new ListNode(3);
	    
	    System.out.println(sol.firstMissing(h));
	    
	    int[][] paint={{1,2,1,0,1},
	    		{1,0,0,1,0},
	    		{1,1,1,1,2},
	    		{0,2,2,1,2}};
	    Point2 p=new Point2(2,2);
//	    sol.floodfill(paint, p, 1, 3);
	    
	    for(int i=0;i<paint.length;i++){
	    	System.out.println(Arrays.toString(paint[i]));
	    }
	    System.out.println();
	    sol.floodFill4(paint, 2,2, 1, 3);
	    
	    for(int i=0;i<paint.length;i++){
	    	System.out.println(Arrays.toString(paint[i]));
	    }
	    
	    
	    TreeLinkNode linkroot=new TreeLinkNode(5);
	    linkroot.left=new TreeLinkNode(3);
	    linkroot.right=new TreeLinkNode(13);
	    linkroot.left.next=linkroot.right;
	    
	    linkroot.left.left=new TreeLinkNode(2);
	    linkroot.left.right=new TreeLinkNode(4);
	    linkroot.right.left=new TreeLinkNode(8);
	    linkroot.left.left.left=new TreeLinkNode(1);
	    linkroot.left.left.next=linkroot.left.right;
	    linkroot.left.right.next=linkroot.right.left;
	    
	    linkroot.right.left.left =new TreeLinkNode(7);
	    linkroot.right.left.right=new TreeLinkNode(11);
	    linkroot.left.left.left.next=linkroot.right.left.left;
	    linkroot.right.left.left.next=linkroot.right.left.right;
	    
	    System.out.println(sol.levelTraversal(linkroot));
	    
	    int[] singles={1,1,2,2,3,3,4,4,5};
	    System.out.println(sol.singleNum(singles));
	    
//	    "world", "hello", "super", "hell" 
	    Set<String> dict=new HashSet<String>();
	    dict.add("world");
	    dict.add("hello");
	    dict.add("super");
	    dict.add("hell");
	    
	    System.out.println(sol.isConcantentationOfOtherStrings(dict, "superheworld"));
	    System.out.println(sol.isConcantentationOfOtherStrings2(dict, "superheworld"));
	    
	    int[] paris={3,4,7,1,2,9,8};
	    sol.indexSumPair(paris);
	    int[] paris2={1,2,3,4,5};
	    System.out.println();
	    sol.indexSumPair(paris2);
	    
	    int[] scores={0,2,3};
	    System.out.println(sol.maxLeadChange(scores, 10, 6));
	    System.out.println(sol.maxLeadChange(scores, 2, 3));
	    System.out.println(sol.maxLeadChange2(scores, 2, 3));
	    System.out.println(sol.solve(10, 6, scores, 0));
	    System.out.println(sol.reverse("the boy ran"));
	    
	    boolean[] git={true,false,false};
	    System.out.println(sol.firstFalse(git));
	    System.out.println(sol.isMatch("ab","a*b"));
	    
	    System.out.println(sol.isMatch2("aa",".*"));
	    System.out.println(sol.longestPalindrome2("ccc"));
	    System.out.println(sol.convertDecToExcel(56));
	    System.out.println(sol.convertExcelToDec("BD"));
	    System.out.println(sol.anaStrStr("car", "actor"));
	    
	    System.out.println(sol.allCombinations(4));
	    int[] arrdup={2,1,3,1,2};
	    System.out.println(sol.removeDuplicates(arrdup));
	    
	    int[] ap={3, 5, 7, 9, 11, 15};
//	    System.out.println(findMissing_binary(ap));
	    System.out.println(sol.firstMissingInAP(ap));
	    System.out.println(sol.getMissingTermInAP(ap));
	    System.out.println(sol.findMissingInAP(ap));
	    
	    System.out.println(sol.RemoveSubstring("ABCSABC", "BC"));
//	    int[] numbers={-2,0,9,-9,1,1,-5,5,-1,1};
	    int[] numbers={-1, -3, 4, 5, 4};
	    sol.continuousSubseq(numbers);
	    
	    HashMap<String, String> rules=new HashMap<String, String>();
	    rules.put("AB", "AA");
	    rules.put("BA", "AA");
	    rules.put("CB", "CC");
	    rules.put("BC", "CC");
	    rules.put("AA", "A");
	    rules.put("CC", "C");
	    System.out.println(sol.simplifyString("ABBCCBBA", rules));
	    
	    System.out.println(sol.canStringBeAPalindrome("abbaddda"));
	    System.out.println(sol.canStringBeAPalindrome2("abbaddda"));
	    
	    
	    System.out.println(sol.isKPalindrome("abxa", 1));
	    System.out.println(sol.isKPalindrome2("abxa", 1));
	    System.out.println(sol.isKPalindrome3("abdxa", 2));
	    
	    String[] strings={"face", "ball", "apple", "art", "ah","alph" };
	    String sortRule="htarfbple";
	    sol.reSorting(strings, sortRule);
	    System.out.println(Arrays.toString(strings));
	    
	    int[] aps={1, 7, 10, 15, 27, 29, 30, 31, 43};
	    System.out.println(sol.findThreeElementsAP(aps));
	    
	    System.out.println(sol.waysOfPerfectSquare(125));
	    
	    int[] test1={-1, 0, 2, 3, 4, 8, 9, 20, 29, 43, 44, 45, 46, 47, 48, 49};

	    System.out.println(sol.FindNearestNum(test1, 7));
	    
	    int[] test2={1,1,2,1,1,3};
	    System.out.println(sol.findAllTriplets(test2));
	    System.out.println(sol.findAllTriplets2(test2));
	    
	    System.out.println(sol.longestCommonSubstring("GeeksforGeeks", "GeeksQuiz"));
	    int[] test3={1,2,3,4,5,6};
	    System.out.println(sol.maxSumNoAdacency(test3));
	    
	    TreeNode r1=new TreeNode(1);
	    r1.left=new TreeNode(2);
	    r1.right=new TreeNode(3);
	    r1.left.left=new TreeNode(4);
	    r1.left.right=new TreeNode(5);
	    r1.left.left.left=new TreeNode(6);
	    r1.left.left.right=new TreeNode(7);
	    
	    sol.inorder(r1);
	    System.out.println();
//	    TreeNode downRoot=sol.upsideDown(r1);
//	    sol.inorder(downRoot);
//	    System.out.println();
	    TreeNode downRoot1=sol.UpsideDownBinaryTreeWithStack(r1);
	    sol.inorder(downRoot1);
	    System.out.println("@@@@@@@@@@@@@@@@");
	    
	    
	    int [ ] sortedCitation = {21, 17, 12, 2, 2, 1}; 
	    System.out.println("The h-index value is: " + sol.getHindexFromSorted(sortedCitation)); 
	    System.out.println(sol.isMatching("aaab", "a*b"));
	    System.out.println(sol.isMatching("aaaabbab", "aa*b*ab+"));
	    System.out.println(sol.isMatching("aaabb", "a+a*b*"));
	    System.out.println(sol.isMatching("aaaabc", "a+aabc"));
	    
	    System.out.println(sol.isMatched("a+a*", ""));
	    
	    char[][] maze={{'.','.','X'},
	    				{'W','.','W'},
	    				{'.','W','.'},
	    				{'O','W','X'}};
	    
	    for(int i=0;i<maze.length;i++){
	    	System.out.println(Arrays.toString(maze[i]));
	    }
	    System.out.println(sol.findPath(maze));
	    
	    System.out.println(sol.stringReduction("bcab"));
	    
	    TreeNode r2=new TreeNode(-2);
	    r2.left=new TreeNode(5);
	    r2.right=new TreeNode(6);
	    r2.left.left=new TreeNode(-8);
	    r2.left.right=new TreeNode(1);
	    r2.left.left.left=new TreeNode(2);
	    r2.left.left.right=new TreeNode(6);
	    
	    r2.right.left=new TreeNode(3);
	    r2.right.right=new TreeNode(9);
	    r2.right.right.right=new TreeNode(0);
	    r2.right.right.right.left=new TreeNode(-4);
	    r2.right.right.right.right=new TreeNode(-1);
	    r2.right.right.right.right.left=new TreeNode(-10);
	    
	    System.out.println(sol.maxTreeLength2(r2));
	    System.out.println("^^^^^^^^^^^^^^vvvvvvvvvvvvv^^^^^^^^^^^^^");
	    System.out.println(sol.maxSumPath(r2));
	    
	    System.out.println(sol.maxRootLeafPathSum(r2));
	    
	    int[] nums1={1,2,3,4,5};
	    System.out.println(Arrays.toString(sol.multiplication(nums1)));
	    System.out.println(Arrays.toString(sol.multiplication2(nums1)));
	    String[] L={"foo", "bar"};
	    System.out.println(sol.findSubstring("arfoothefoobarman", L));
	    
	    System.out.println(sol.minSteps("abcdc"));
	    
	    int[] zero_ones={0, 1, 0, 0, 1, 1, 1, 1, 0 };
	    sol.intervalWithEqual0sAnd1s(zero_ones);
	    System.out.println();
	    int[] nums3={1,-1,2,-1,-1,1,-1};
	    sol.continuousSubseq(nums3);
	    System.out.println();
	    sol.subarrayEqualsZero(nums3);
	    String str = "    c   d ";
	    System.out.println(sol.countWords(str));
	    
	    System.out.println(sol.atof("-33.921"));
	    
	    System.out.println(sol.numOfPalindromes("abba"));
	    
	    int[] t={1, 2, 3, 8, 10, 5, 6, 7, 12, 9, 4, 0};
	    System.out.println(sol.LeastRemoval(t));
	    
	    int[] seq={1, 4};
	    System.out.println(sol.subSequenceSumToTotal(seq, 0));
	    int[] seq2={-1, -4, 1, 0, -2, -3, 7};
	    System.out.println(sol.subSequenceSumToTotal2(seq2, 2));
	    
	    
	    System.out.println();
	    ListNode ll=new ListNode(2);
	    ListNode ll1=new ListNode(1);
	    ListNode ll2=new ListNode(5);
	    ListNode ll3=new ListNode(2);
	    ListNode ll4=new ListNode(4);
	    ListNode ll5=new ListNode(2);
	    ListNode ll6=new ListNode(2);
	    ListNode ll7=new ListNode(3);
	    ListNode ll8=new ListNode(1);
	    ListNode ll9=new ListNode(2);
	    ll.next=ll1;
	    ll1.next=ll2;
	    ll2.next=ll3;
	    ll3.next=ll4;
	    ll4.next=ll5;
	    ll5.next=ll6;
	    ll6.next=ll7;
	    ll7.next=ll8;
	    ll8.next=ll9;
	    
	    ListNode llhead=sol.removeNodes(ll, 2);
	    while(llhead!=null){
	    	System.out.print(llhead.val+" ");
	    	llhead=llhead.next;
	    }
	    System.out.println();
	    
	    int t5[] = {12, 3, 5, 7, 4, 19, 26};
	    System.out.println(sol.findKthSmallest(t5, 0));
	    
	    ListNode head1=new ListNode(1);
	    ListNode head2=new ListNode(0);
	    
	    ListNode listnode1=new ListNode(3);
	    ListNode listnode2=new ListNode(30);
	    ListNode listnode3=new ListNode(90);
	    ListNode listnode4=new ListNode(110);
	    ListNode listnode5=new ListNode(120);
	    
	    ListNode listnode6=new ListNode(3);
	    ListNode listnode7=new ListNode(12);
	    ListNode listnode8=new ListNode(32);
	    ListNode listnode9=new ListNode(90);
	    ListNode listnode10=new ListNode(100);
	    ListNode listnode11=new ListNode(120);
	    ListNode listnode12=new ListNode(130);
	    
	    
	    head1.next=listnode1;
	    listnode1.next=listnode2;
	    listnode2.next=listnode3;
	    listnode3.next=listnode4;
	    listnode4=listnode5;
	    
	    head2.next=listnode6;
	    listnode6.next=listnode7;
	    listnode7.next=listnode8;
	    listnode8.next=listnode9;
	    listnode9.next=listnode10;
	    listnode10.next=listnode11;
	    listnode11.next=listnode12;
	    
	    int[] seq1={ 1,7,4,9,2,5 };

	    System.out.println(sol.longestZigZag(seq1));
	    int[] coin={1,3,5};
	    System.out.println(sol.minChange(coin,10));
	    
	    int[] donations={10, 3, 2, 5, 7, 8 };
	    System.out.println(sol.badNeighbors(donations));
	    System.out.println(sol.getNthUglyNum(7));
	    
	    int[] lmh={10,2,8,-1,18,7,11,0,-3,9};
	    sol.sortLowMedHight(lmh);
	    System.out.println(Arrays.toString(lmh));
	    
	    int[] kRanks={5,2,3,1,7,10,8,4,6,9,7};
	    sol.sortKRanks(kRanks);
	    System.out.println(Arrays.toString(kRanks));
	    
	    int array[] = { 1,7,10, 50,19};
	    System.out.println(sol.findMax(array));
	    System.out.println(sol.longestValidParentheses("(()"));
	    
	    System.out.println(sol.validParenthesis("()()(())(((())))()()"));
	    System.out.println(sol.divide2(12, -3));
	    
	    int[] arrwithdups={1,2,2};
	    System.out.println(sol.removeDuplicates2(arrwithdups));
	    int[] zeroOne={0,0,0,0,0,1};
	    System.out.println(sol.findFirstOne(zeroOne));
	    
	    char[][] mat={{'1','1'},{'1','1'},{'0','1'}};
	    System.out.println(sol.maximalRectangle(mat));
	    
	    System.out.println(sol.numDecodings("10234"));
	    System.out.println(sol.decoding("10234"));
	}
}
