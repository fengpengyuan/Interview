package facebook;

import java.util.Stack;

public class NextInorderSuccessorIterator {
	Stack<TreeNode> stk=new Stack<TreeNode>();
	TreeNode root;
	
	public NextInorderSuccessorIterator(TreeNode root){
		this.root=root;
	}
	
	public TreeNode next(){
		TreeNode cur=root;
		while(cur!=null){
			stk.push(cur);
			cur=cur.left;
		}
		cur=stk.pop();
		root=cur;
		root=cur.right;
		return cur;
	}
	
	public boolean hasNext(){
		if(root!=null||!stk.isEmpty())
			return true;
		return false;
	}
	
	public static void main(String[] args){
		TreeNode root=new TreeNode(5);
		root.left=new TreeNode(2);
		root.left.left=new TreeNode(1);
		root.left.right=new TreeNode(3);
		
		root.right=new TreeNode(8);
		root.right.right=new TreeNode(10);
		
		NextInorderSuccessorIterator it=new NextInorderSuccessorIterator(root);
		
		while(it.hasNext()){
			System.out.print(it.next().val+" ");
		}
	}
}
