package com.lintcode;

public class SegmentTree {
	SegmentTreeNode root;
    int[] A;
    
    public SegmentTree(int[] A) {
        // write your code here
        this.A=A;
        root=build(A, 0, A.length-1);
    }
    
    public SegmentTreeNode build(int[] A, int start, int end){
//        if(start>end)
//            return null;
//        if(start==end)
//            return new SegmentTreeNode(start, end, A[start]);
//        long sum=0;
//        for(int i=start;i<=end;i++)
//            sum+=A[i];
//        SegmentTreeNode root=new SegmentTreeNode(start, end, sum);
//        root.left=build(A, start, (start+end)/2);
//        root.right=build(A, (start+end)/2+1, end);
//        return root;
    	SegmentTreeNode root=new SegmentTreeNode(start, end, A[start]);
        root.left=build(A, start, (start+end)/2);
        root.right=build(A, (start+end)/2+1, end);
        root.sum=root.left.sum+root.right.sum;
        return root;
    }
    
    /**
     * @param start, end: Indices
     * @return: The sum from start to end
     */
    public long query(int start, int end) {
        // write your code here
        return query(root, start, end);
    }
    
    public long query(SegmentTreeNode root, int start, int end){
        if(root.start>=start&&root.end<=end)
            return root.sum;
        if(start>root.end||end<root.start)
            return 0;
        return query(root.left, start, end)+query(root.right, start, end);
    }
    
    /**
     * @param index, value: modify A[index] to value.
     */
    public void modify(int index, int value) {
        // write your code here
        modify(root, index, value);
    }
    
    public void modify(SegmentTreeNode root, int index, int value){
        if(root==null)
            return;
        if(index<root.start||index>root.end)
            return;
        
        if(index<=(root.start+root.end)/2)
            modify(root.left, index, value);
        else
            modify(root.right, index, value);
        if(root.start==index&&root.end==index){
            root.sum=value;
            A[index]=value;
        }
        else
            root.sum=root.left.sum + root.right.sum+(value-A[index]);
    }
}
