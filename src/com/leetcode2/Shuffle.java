package com.leetcode2;

import java.util.Random;

public class Shuffle {
	int[] original;
    public Shuffle(int[] nums) {
        original = nums;
    }
    
    /** Resets the array to its original configuration and return it. */
    public int[] reset() {
        return original;
    }
    
    /** Returns a random shuffling of the array. */
    public int[] shuffle() {
        int[] res = original.clone();
        for(int i=0;i<res.length;i++){
            int j=new Random().nextInt(i+1);
            swap(res, i, j);
        }
        return res;
    }
    
    public void swap(int[] a, int i, int j){
        int t=a[i];
        a[i]=a[j];
        a[j]=t;
    }
}
