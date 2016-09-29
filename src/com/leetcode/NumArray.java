package com.leetcode;

public class NumArray {
	int[] nums;
	int[] t;
    public NumArray(int[] nums) {
        this.nums = nums;
        if(nums==null||nums.length==0)
        	return;
        t=new int[nums.length];
        t[0]=nums[0];
        for(int i=1;i<nums.length;i++){
        	t[i]=t[i-1]+nums[i];
        }
    }

    void update(int i, int val) {
        nums[i]=val;
        for(int k=i;k<nums.length;k++){
        	t[i]=t[i]+val-nums[i];
        }
    }

    public int sumRange(int i, int j) {
    	return t[j]-t[i]+nums[i];
    }
}
