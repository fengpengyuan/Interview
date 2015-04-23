package com.lintcode;

import java.util.Stack;

public class MinStack {
	Stack<Integer> stk1;
    Stack<Integer> stk2;
    public MinStack() {
        // do initialize if necessary
        stk1=new Stack<Integer>();
        stk2=new Stack<Integer>();
    }

    public void push(int number) {
        // write your code here
        stk1.push(number);
        if(stk2.isEmpty()||number<=stk2.peek())
            stk2.push(number);
    }

    public int pop() {
        // write your code here
        int top=stk1.pop();
        if(top==stk2.peek())
            stk2.pop();
        return top;
    }

    public int min() {
        // write your code here
        return stk2.peek();
    }
}
