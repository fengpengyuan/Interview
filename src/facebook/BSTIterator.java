package facebook;

import java.util.Stack;

public class BSTIterator {
	Stack<TreeNode> stk;
    public BSTIterator(TreeNode root) {
        stk=new Stack<TreeNode>();
        while(root!=null){
            stk.push(root);
            root=root.left;
        }
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return !stk.isEmpty();
    }

    /** @return the next smallest number */
    public int next() {
        if(!hasNext())
            return Integer.MIN_VALUE;
        TreeNode top=stk.pop();
        int res=top.val;
        top=top.right;
        while(top!=null){
            stk.push(top);
            top=top.left;
        }
        return res;
    }
}

/**
 * Your BSTIterator will be called like this:
 * BSTIterator i = new BSTIterator(root);
 * while (i.hasNext()) v[f()] = i.next();
 */