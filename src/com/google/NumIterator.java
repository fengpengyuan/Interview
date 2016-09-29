/**
 * 
 */
package com.google;

import java.util.Arrays;
import java.util.List;

/**
 * @author fengpeng
 *
 */
/*
 * 实现一个iterator, input 是一个array{3, 8, 0, 12, 2, 9}, 希望输出是 {8, 8, 8, 9, 9},
 * 也就是eventh number代表 词频， oddth number 代表词， {3, 8, 12, 0, 2, 9}， 就是3个8，
 * 0个12， 2个9. 
 * 和美眉商量了输入不用array, 用个List<Integer> 简单好多
 */
 
public class NumIterator {
	int[] nums;
	int freq;
	int curPos;
	public NumIterator(int[] nums){
		this.nums = nums;
		if(nums.length==0||nums.length%2!=0)
			return;
		curPos=0;
		freq = nums[curPos];
	}
	
	public boolean hasNext(){
		if(freq!=0)
			return true;
		curPos+=2;
		while(curPos<nums.length&&nums[curPos]==0){
			curPos+=2;
		}
		if(curPos>=nums.length)
			return false;
		freq = nums[curPos];
		return true;
	}
	
	public int next(){
		int res = nums[curPos+1];
		freq--;
//		System.out.println(freq+" "+curPos);
		return res;
	}
	
	
	public static void main(String[] args){
		int[] nums={0,8,0,12,2,9, 0,1,0,2,1,3, 0,2,3,1};
//		List<Integer> numbers = Arrays.asList(nums);
		NumIterator ni=new NumIterator(nums);
		
		while(ni.hasNext()){
			System.out.print(ni.next()+" ");
//			System.out.println(curPos);
		}
	}
}
