package facebook;

/*
 * implement iterator (hasNext, next) for two BSTs，就是给两棵BST写一个iterator，每次取出最小值
 next1, next2分别记录BST1， BST2的当前最小值，返回Math.min(next1, next2)后更新next1，next2.
 注意hasNext()的写法，因为两颗树都空了之后还要检查next1，next2当中是否还有值。
 */
public class TwoBSTIterator {
	BSTIterator bi1;
	BSTIterator bi2;
	Integer next1;
	Integer next2;
	public TwoBSTIterator(TreeNode root1, TreeNode root2){
		bi1=new BSTIterator(root1);
		bi2=new BSTIterator(root2);
		next1=bi1.next();
		next2=bi2.next();
	}
	
	public boolean hasNext(){
		if(!bi1.hasNext()&&!bi2.hasNext()){
			return next1!=null||next2!=null;
		}
		return true;
	}
	
	public int next(){
		if(next1==null){
			if(next2==null)
				return Integer.MAX_VALUE;
			int rs=next2;
			next2=null;
			return rs;
		}
		if(next2==null){
			if(next1==null)
				return Integer.MAX_VALUE;
			int rs=next1;
			next1=null;
			return rs;
		}
		if(next1<next2){
			int rs=next1;
			next1=bi1.next();
//			System.out.println("Tree 1");
			return rs;
		}else{
			int rs=next2;
			next2=bi2.next();
//			System.out.println("Tree 2");
			return rs;
		}
	}
	
	public static void main(String[] args){
		TreeNode root1 = new TreeNode(10);
		root1.left = new TreeNode(5);
		root1.left.right = new TreeNode(8);
		root1.left.left = new TreeNode(2);

		root1.right = new TreeNode(13);
		root1.right.left = new TreeNode(11);
		
		TreeNode root = new TreeNode(9);
		root.left = new TreeNode(6);
		root.left.right = new TreeNode(7);
//		 root.left.right.left = new TreeNode(3);
		root.left.left = new TreeNode(1);

		root.right = new TreeNode(12);
		
		root.right.right = new TreeNode(14);
		
		TwoBSTIterator twobi=new TwoBSTIterator(root1, root);
		while(twobi.hasNext()){
			System.out.print(twobi.next()+" ");
		}
	}
}
