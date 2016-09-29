package com.leetcode2;

public class NumArray2 {
	int[] nums;
	int[] sumArr;
	public NumArray2(int[] nums) {
		this.nums = nums;
		this.sumArr = new int[nums.length+1];
		for(int i=0;i<nums.length;i++){
			add(i+1, nums[i]);
		}
	}
	
	public void add(int x, int val){
		while(x<=nums.length){
			sumArr[x]+=val;
			x+=lowbit(x);
		}
	}
	
	public int lowbit(int x){
		return x&(-x);
	}
	
	void update(int i, int val) {
		add(i+1, val);
		nums[i] = val;
	}
	
	public int sumRange(int i, int j) {
		return getSum(j+1)-getSum(i);
	}
	
	public int getSum(int i){
		int res = 0;
		while(i>0){
			res+=sumArr[i];
			i-=lowbit(i);
		}
		return res;
	}
}
