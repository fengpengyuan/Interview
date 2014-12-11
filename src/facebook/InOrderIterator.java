package facebook;

import java.util.Stack;

public class InOrderIterator {
	private TreeNode tree;

    private TreeNode current;
    private TreeNode next;
    private boolean done;
    private Stack<TreeNode> stack;

    public InOrderIterator(TreeNode tree) {
        this.tree = tree;
        this.stack = new Stack<TreeNode>();
    }

    public void begin() {
        this.stack.clear();
        this.current = null;
        this.next = this.tree;
        this.done = false;

        this.next();
    }

    public boolean end() {
        return this.done;
    }

    public void next() {
        this.current = this.next;
        while (current != null) {
            stack.push(current);
            current = current.left;
        }
    
        if (!stack.isEmpty()) {
            this.current = stack.pop();
            this.next = current.right;
        } else {
            this.done = true;
        }
    }

    public int current() {
        return current.val;
    }
    
    public static void main(String[] args) {
    	TreeNode r=new TreeNode(8);
	    r.left=new TreeNode(5);
	    r.right=new TreeNode(12);
	    r.left.right=new TreeNode(7);
	    r.right.left=new TreeNode(9);
	    r.right.left.right=new TreeNode(11);
        InOrderIterator it = new InOrderIterator(r);
        for (it.begin(); !it.end(); it.next()) {
            System.out.println(it.current());
        }
    }

}
