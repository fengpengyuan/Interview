package com.leetcode2;

public class NumArray_mutable {
	public class SegmentTreeNode {
		int start;
		int end;
		SegmentTreeNode left, right;
		int sum;

		public SegmentTreeNode(int start, int end) {
			this.start = start;
			this.end = end;
			this.sum = 0;
			this.left = this.right = null;
		}
	}

	SegmentTreeNode root = null;

	public NumArray_mutable(int[] nums) {
		root = buildTree(nums, 0, nums.length - 1);
	}

	public SegmentTreeNode buildTree(int[] nums, int start, int end) {
		if (start > end) {
			return null;
		} 
			SegmentTreeNode node = new SegmentTreeNode(start, end);
			if (start == end)
				node.sum = nums[start];
			else {
				int mid = (start + end) / 2;
				node.left = buildTree(nums, start, mid);
				node.right = buildTree(nums, mid + 1, end);
				node.sum = node.left.sum + node.right.sum;
			}
			return node;
		
	}

	void update(int i, int val) {
		update(root, i, val);
	}

	void update(SegmentTreeNode root, int i, int val) {
		if (root.start == root.end)
			root.sum = val;
		else {
			int mid = (root.start + root.end) / 2;
			if (i <= mid)
				update(root.left, i, val);
			else
				update(root.right, i, val);
			root.sum = root.left.sum + root.right.sum;
		}
	}

	public int sumRange(int i, int j) {
		return sumRange(root, i, j);
	}

	public int sumRange(SegmentTreeNode root, int i, int j) {
		if(root==null)
			return 0;
		if (root.start == i && root.end == j)
			return root.sum;
		int mid = (root.start + root.end) / 2;
		if (j <= mid)
			return sumRange(root.left, i, j);
		if (i > mid)
			return sumRange(root.right, i, j);
		else
			return sumRange(root.left, i, mid)
					+ sumRange(root.right, mid + 1, j);
	}

	public static void main(String[] args) {
		int[] nums = { 1, 3, 5, 7};

		NumArray_mutable numArr = new NumArray_mutable(nums);
		System.out.println(numArr.sumRange(1, 2));
	}
}
