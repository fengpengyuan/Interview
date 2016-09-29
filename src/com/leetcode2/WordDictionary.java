package com.leetcode2;

public class WordDictionary {
TrieNode root;
    
    public WordDictionary(){
        root=new TrieNode();
    }

    // Adds a word into the data structure.
    public void addWord(String word) {
        TrieNode node = root;
        for(int i=0;i<word.length();i++){
            TrieNode[] children=node.children;
            char c = word.charAt(i);
            if(children[c-'a']!=null){
                node=children[c-'a'];
            } else {
                node=new TrieNode(c);
                children[c-'a']=node;
            }
        }
        node.isLeaf=true;
    }

    // Returns if the word is in the data structure. A word could
    // contain the dot character '.' to represent any one letter.
    public boolean search(String word) {
        return dfsSearch(root, word, 0);
    }
    
    public boolean dfsSearch(TrieNode node, String word, int cur){
        if(node==null||cur==word.length()&&!node.isLeaf)
            return false;
        if(cur==word.length()&&node.isLeaf)
            return true;
        TrieNode[] children=node.children;
        char c=word.charAt(cur);
        if(c=='.'){
            for(int i=0;i<26;i++){
                boolean res=dfsSearch(children[i], word, cur+1);
                if(res)
                    return true;
            }
            return false;
        } else {
            node=children[c-'a'];
            return dfsSearch(node, word, cur+1);
        }
    }
}

