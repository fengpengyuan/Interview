package com.leetcode2;

import java.util.Stack;

public class BSTPostOrderIterator {
	Stack<TreeNode> stk;
	TreeNode pre;

    public BSTPostOrderIterator(TreeNode root) {
    	pre=null;
        stk=new Stack<TreeNode>();
        TreeNode cur=root;
        while(cur!=null){
            stk.push(cur);
            cur=cur.left;
        }
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return !stk.isEmpty();
    }

    /** @return the next smallest number */
	public int next() {
		while (true) {
			TreeNode top = stk.peek();
			int val = top.val;
			if (top.right != null && pre != top.right) {
				top = top.right;
				while (top != null) {
					stk.push(top);
					top = top.left;
				}
			} else {
				pre = stk.pop();
				return val;
			}
		}
	}
    
    public static void main(String[] strs){
    	TreeNode root=new TreeNode(10);
		root.left=new TreeNode(6);
		root.left.left=new TreeNode(1);
		root.left.right=new TreeNode(2);
		root.left.left.right=new TreeNode(30);
		root.right=new TreeNode(30);
		root.right.right=new TreeNode(33);
		
		BSTPostOrderIterator iterator=new BSTPostOrderIterator(root);
		while(iterator.hasNext()){
			int val=iterator.next();
			System.out.print(val+" ");
		}
    }
}
