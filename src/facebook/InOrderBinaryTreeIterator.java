package facebook;

import java.util.NoSuchElementException;
import java.util.Stack;

public class InOrderBinaryTreeIterator {
	Stack<TreeNode> stack = new Stack<TreeNode>();
	TreeNode current;

	/** Push node cur and all of its left children into stack */
	private void pushLeftChildren(TreeNode cur) {
		while (cur != null) {
			stack.push(cur);
			cur = cur.left;
		}
	}

	public InOrderBinaryTreeIterator(TreeNode root) {
		pushLeftChildren(root);
	}

	public boolean hasNext() {
		return !stack.isEmpty();
	}

	public void next() {
		if (!hasNext()) {
			throw new NoSuchElementException("All nodes have been visited!");
		}

		TreeNode res = stack.pop();
		current = res;
		pushLeftChildren(res.right);

	}

	public int value() {
		return current.val;
	}

	public static void main(String[] args) {
		TreeNode r = new TreeNode(8);
		r.left = new TreeNode(5);
		r.right = new TreeNode(12);
		r.left.right = new TreeNode(7);
		r.right.left = new TreeNode(9);
		r.right.left.right = new TreeNode(11);
		InOrderBinaryTreeIterator it = new InOrderBinaryTreeIterator(r);
		while(it.hasNext()){
			it.next();
			System.out.print(it.value()+" ");
		}

	}
}
