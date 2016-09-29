package com.lintcode;

import java.util.HashMap;

public class WordDictionary {
TrieNode root;
    
    public WordDictionary(){
        root=new TrieNode();
    }

    // Adds a word into the data structure.
    public void addWord(String word) {
        // Write your code here
        if(word.length()==0)
            return;
        TrieNode cur=root;
        for(int i=0;i<word.length();i++){
            char c=word.charAt(i);
            HashMap<Character, TrieNode> children=cur.children;
            if(children.containsKey(c))
                cur=children.get(c);
            else{
                TrieNode node=new TrieNode(c);
                children.put(c, node);
                cur=node;
            }
        }
        cur.isLeaf=true;
    }

    // Returns if the word is in the data structure. A word could
    // contain the dot character '.' to represent any one letter.
    public boolean search(String word) {
        // Write your code here
        TrieNode cur=root;
        return search(word, cur);
    }
    
    public boolean search(String word, TrieNode root){
    	for(int i=0;i<word.length();i++){
            char c=word.charAt(i);
            HashMap<Character, TrieNode> children=root.children;
            if(children.containsKey(c))
                root=children.get(c);
            else if(c=='.'){
                for(char key: children.keySet()){
                    root=children.get(key);
                    if(search(word.substring(i+1), root))
                        return true;
                }
                return false;
            }else
                return false;
        }
        if(root.isLeaf)
            return true;
        return false;
    }
    
    public static void main(String[] args){
    	WordDictionary wd=new WordDictionary();
    	wd.addWord("bad");
    	wd.addWord("dad");
    	wd.addWord("mad");
    	
    	System.out.println(wd.search("pad"));
    	System.out.println(wd.search(".ad"));
    	System.out.println(wd.search("b.."));
    }
}

// Your WordDictionary object will be instantiated and called as such:
// WordDictionary wordDictionary = new WordDictionary();
// wordDictionary.addWord("word");
// wordDictionary.search("pattern");