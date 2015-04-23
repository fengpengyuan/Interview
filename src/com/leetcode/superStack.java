package com.leetcode;

import java.util.Stack;

public class superStack {
	Stack<Integer> stk;
	public superStack(){
		stk=new Stack<Integer>();
	}
	
	public void push(int x){
		stk.push(x);
	}
	
	public int pop(){
		if(!stk.isEmpty())
			return stk.pop();
		return -1;
	}
	
	public boolean isEmpty(){
		if(stk.isEmpty())
			return true;
		return false;
	}
	
	public void increment(int a, int b){
		if(stk.isEmpty())
			return;
		Stack<Integer> temp=new Stack<Integer>();
		while(!stk.isEmpty()){
			temp.push(stk.pop());
		}
		for(int i=0;i<a;i++){
			if(!temp.isEmpty())
				stk.push(temp.pop()+b);
			else
				break;
		}
		while(!temp.isEmpty())
			stk.push(temp.pop());
	}
	
	public static void main(String[] args){
		superStack stack=new superStack();
		stack.push(10);
		stack.push(10);
		stack.push(10);
		stack.push(10);
		stack.push(10);
		stack.push(10);
		stack.push(10);
		
		stack.increment(5, 20);
		
		while(!stack.isEmpty()){
			System.out.print(stack.pop()+" ");
		}
	}
}
