package com.lintcode;

import java.util.Stack;

public class TreeIterator {
	Stack<TreeNode> stk=new Stack<TreeNode>();
    public TreeIterator(TreeNode root) {
        // write your code here
        while(root!=null){
            stk.push(root);
            root=root.left;
        }
    }

    //@return: True if there has next node, or false
    public boolean hasNext() {
        // write your code here
        return !stk.isEmpty();
    }
    
    //@return: return next node
    public TreeNode next() {
        // write your code here
        TreeNode top=stk.pop();
        if(top.right!=null){
            TreeNode node=top.right;
            while(node!=null){
                stk.push(node);
                node=node.left;
            }
        }
        return top;
    }
}
