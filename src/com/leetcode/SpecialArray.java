package com.leetcode;

public class SpecialArray {
	char[] arr;
	
	public SpecialArray(int n){
		arr=new char[n];
	}
	
	public int get(int i){
		return arr[i];
	}
	
	public char swapA(int i){
		char t=arr[i];
		arr[i]='a';
		return t;
	}
	
	public void sort(){
		
	}
}
