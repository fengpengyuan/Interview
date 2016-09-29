package com.lintcode;

import java.util.ArrayList;

public class SegmentTreeNode {
	public int start, end, max;
	public long sum;
	public SegmentTreeNode left, right;

	public SegmentTreeNode(int start, int end) {
		this.start = start;
		this.end = end;
		this.left = this.right = null;
	}

	public SegmentTreeNode(int start, int end, int max) {
		this.start = start;
		this.end = end;
		this.max = max;
		this.left = this.right = null;
	}
	
	public SegmentTreeNode(int start, int end, long sum) {
		this.start = start;
		this.end = end;
		this.sum = sum;
		this.left = this.right = null;
	}

	public ArrayList<Integer> intervalMinNumber(int[] A,
			ArrayList<Interval> queries) {
		// write your code here
		ArrayList<Integer> res = new ArrayList<Integer>();
		SegmentTreeNode root = build(A, 0, A.length - 1);
		for (int i = 0; i < queries.size(); i++) {
			Interval interval = queries.get(i);
			int queryRs = query(root, interval.start, interval.end);
			res.add(queryRs);
		}
		return res;
	}

	public SegmentTreeNode build(int[] A, int start, int end) {
		// write your code here
		if (end < start)
			return null;
		if (start == end)
			return new SegmentTreeNode(start, end, A[start]);
		int max = A[start];
		for (int i = start + 1; i <= end; i++)
			max = Math.min(A[i], max);
		SegmentTreeNode root = new SegmentTreeNode(start, end, max);
		root.left = build(A, start, (start + end) / 2);
		root.right = build(A, (start + end) / 2 + 1, end);
		return root;
	}
	
	public SegmentTreeNode build2(int[] A, int start, int end){
        if(start>end)
            return null;
        if(start==end){
            return new SegmentTreeNode(start, end, A[start]);
        }
        
        SegmentTreeNode root=new SegmentTreeNode(start, end, A[end]);
        int mid=(start+end)/2;
        root.left=build(A, start, mid);
        root.right=build(A, mid+1, end);
        if(root.left!=null&&root.left.max>root.max)
            root.max=root.left.max;
        if(root.right!=null&&root.right.max>root.max)
            root.max=root.right.max;
        return root;
    }

	public int query(SegmentTreeNode root, int start, int end) {
		if (root.start >= start && root.end <= end)
			return root.max;
		if (start > root.end || end < root.start)
			return Integer.MIN_VALUE;
		return Math.max(query(root.left, start, end),
				query(root.right, start, end));
	}
}
