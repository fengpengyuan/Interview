package com.leetcode2;

public class TreeNode {
	int val;
	TreeNode left;
	TreeNode right;
	int size;

	TreeNode(int x) {
		val = x;
		size=1;
	}
	
	public static TreeNode insert(TreeNode root, int val){
		if(root==null)
			return new TreeNode(val);
		if(val<root.val)
			root.left=insert(root.left, val);
		else if(val>root.val)
			root.right=insert(root.right, val);
		root.size++;
		return root;
	}
	
	public static int rank(TreeNode root, int val) {
        if (root==null) return 0;
        if (val<root.val) return rank(root.left,val);
        if (val==root.val) return size(root.left);
        return root.size-size(root.right)+rank(root.right,val);
    }
    private static int size(TreeNode root) {
        if (root==null) return 0;
        return root.size;
    }
//	public static TreeNode insert(TreeNode root, int val){
//		if(root==null)
//			return new TreeNode(val);
//		if(val<root.val)
//			root.left=insert(root.left, val);
//		else if(val>root.val)
//			root.right=insert(root.right, val);
//		return root;
//	}
//	
//	public static int getRank(TreeNode root, int val){
//		if(root==null)
//			return 0;
//		int rank=0;
//		while(root!=null){
//			if(val<root.val)
//				root=root.left;
//			else if(val>root.val){
//				rank+=1+getCount(root.left);
//				root=root.right;
//			} else
//				return rank+getCount(root.left);
//		}
//		return 0;
////		if(val==root.val)
////			return getCount(root.left);
////		else if(val<root.val){
////			return getRank(root.left, val);
////		} else
////			return getCount(root.left)+getRank(root.right, val)+1;
//		
//	}
//	
//	public static int getCount(TreeNode root){
//		if(root==null)
//			return 0;
//		return getCount(root.left)+getCount(root.right)+1;
//	}
}