package com.leetcode;

import java.util.Stack;

public class MinStack {
	Stack<Integer> stk1=new Stack<Integer>();
	Stack<Integer> stk2=new Stack<Integer>();
	public void push(int x) {
        stk1.push(x);
        if(stk2.isEmpty()||x<=stk2.peek())
        	stk2.push(x);
    }

    public void pop() {
        int t=stk1.pop();
        if(t==stk2.peek())
        	stk2.pop();
    }

    public int top() {
        return stk1.peek();
    }

    public int getMin() {
        return stk2.peek();
    }
    
    public static void main(String[] args){
    	
//    	push(2147483646),push(2147483646),push(2147483647),
//    	top,pop,getMin,pop,getMin,pop,push(2147483647),top,getMin,push(-2147483648),top,getMin,pop,getMin
    	MinStack mstk=new MinStack();
    	mstk.push(-2);
    	mstk.push(0);
    	mstk.push(-1);
    	System.out.println(mstk.stk1.size()+" "+mstk.stk2.size());
    	System.out.println(mstk.getMin());
    	System.out.println(mstk.top());
    	mstk.pop();
    	System.out.println(mstk.getMin());
//    	mstk.pop();
//    	mstk.getMin();
//    	mstk.pop();
//    	mstk.push(2147483647);
//    	mstk.top();
//    	System.out.println(mstk.getMin());
    }
}
