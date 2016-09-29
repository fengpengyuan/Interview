package com.leetcode2;

import java.util.ArrayList;
import java.util.List;
import java.util.Stack;

/*
 * 实现一个mini parser, 输入是以下格式的string:"324" or"[123,456,[788,799,833],[[]],10,[]]" 
 * 要求输出:324 or [123,456,[788,799,833],[[]],10,[]].
 */
public class NestedIntList {
	int val;
	List<NestedIntList> intList;
	boolean isNum;
	
	public NestedIntList(int val){
		this.val=val;
		this.isNum=true;
	}
	
	public NestedIntList(){
		this.intList=new ArrayList<NestedIntList>();
		this.isNum=false;
	}
	
	public void add(NestedIntList lst){
		intList.add(lst);
	}
	
	public String toString() {
		if(isNum)
			return String.valueOf(val);
		return intList.toString();
	}
	
	public NestedIntList miniParser(String s){
		if(s==null||s.length()==0)
			return null;
		Stack<NestedIntList> stk=new Stack<NestedIntList>();
		int i=0, left=0;
		NestedIntList res=null;
		while(i<s.length()){
			char c=s.charAt(i);
			if(c=='['){
				NestedIntList num=new NestedIntList();
				if(!stk.isEmpty())
					stk.peek().add(num);
				stk.push(num);
				left=i+1;
			}else if(c==','||c==']'){
				if(left!=i){
					int val=Integer.parseInt(s.substring(left, i));
					NestedIntList num=new NestedIntList(val);
					stk.peek().add(num);
				}
				left=i+1;
				if(c==']'){
					res=stk.pop();
				}
			}
			i++;
		}
		if(left!=i){
			res=new NestedIntList(Integer.parseInt(s));
		}
		return res;
	}
	
	public static void main(String[] args){
		String s="[123,456,[788,799,833],[[]],10,[]]";
		String s1="123";
		NestedIntList nil=new NestedIntList();
		NestedIntList res = nil.miniParser(s);
		System.out.println(res.toString());
		
		NestedIntList res1 = nil.miniParser(s1);
		System.out.println(res1.toString());
	}
}
