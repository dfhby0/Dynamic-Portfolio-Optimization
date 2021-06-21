#include<bits/stdc++.h>
using namespace std;
bool check(string s,int l, int r)
{
    //先判头符号
    int t = l;
    bool isint = false;
    bool intcount = false;
    if(s[t] == '+'||s[t] == '-')t ++;
    if(s[t] == 'e'||s[t] == 'E')isint = true,t++;
    while(isdigit(s[t]) && t <= r)intcount = true,t ++;
    if(isint && t <= r)return false;
    if(t <= r && s[t] == '.')t ++;
    while(isdigit(s[t]) && t <= r)intcount = true,t ++;
    if(t <= r)return false;
    if(~intcount)return false;
    return true;
}

int main()
{
    string s;
    s = '.1';
    if(check(s,0,1))cout << 1;
    else cout << 0;
    return 0;
}