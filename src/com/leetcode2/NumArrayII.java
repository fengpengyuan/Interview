package com.leetcode2;

public class NumArrayII {
	int[] BIT;
    int[] nums;
    int size;

    public NumArrayII(int[] nums) {
        this.nums=nums;
        this.size=nums.length;
        BIT=new int[size+1];
        for(int i=0;i<size;i++){
            updateBIT(i, nums[i]);
        }
    }
    
    public void updateBIT(int i, int val){
        int index=i+1;
        while(index<=size){
            BIT[index]+=val;
            index+=index&(-index);
        }
    }

    void update(int i, int val) {
        int delta=val-nums[i];
        updateBIT(i, delta);
        nums[i]=val;
    }
    
    public int getSum(int i){
        int sum=0;
        i=i+1;
        while(i>0){
            sum+=BIT[i];
            i-=(i&-i);
        }
        return sum;
    }

    public int sumRange(int i, int j) {
        if(i==0)
            return getSum(j);
        return getSum(j)-getSum(i-1);
    }
}
