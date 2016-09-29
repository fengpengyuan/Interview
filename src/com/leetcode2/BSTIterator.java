package com.leetcode2;

import java.util.Stack;

public class BSTIterator {
	Stack<TreeNode> stk;

    public BSTIterator(TreeNode root) {
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
        TreeNode top=stk.pop();
        int val=top.val;
        if(top.right!=null){
            top=top.right;
            while(top!=null){
                stk.push(top);
                top=top.left;
            }
        }
        return val;
    }
}
