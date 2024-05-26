#pragma once
#define Row 1000
#define Columns 1000

class CMFCMemoView : public CScrollView
{
protected: // serialization에서만 만들어집니다.
    CMFCMemoView() noexcept;
    DECLARE_DYNCREATE(CMFCMemoView)

public:
    CMFCMemoDoc* GetDocument() const;

public:
    virtual void OnDraw(CDC* pDC);  // 이 클래스에 대한 그리기입니다.
    virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
    virtual void OnInitialUpdate();

protected:

public:
    virtual ~CMFCMemoView();

#ifdef _DEBUG
    virtual void AssertValid() const;
    virtual void Dump(CDumpContext& dc) const;
#endif

private:
    // *********************************** 캐럿 ***********************************
    void UpdateCaretPosition();
    void CreateCaretIfNeeded();
    // *********************************** 캐럿 ***********************************

    int m_nScrollWidth;
    int m_nScrollHeight;
    int m_nPageSize; 

protected:

    DECLARE_MESSAGE_MAP()
public:

    wchar_t m_textArray[Row][Columns];      // 텍스트를 저장할 2차원 배열

    int m_nCurrentRow;                      // 현재 행
    int m_nCurrentColumn;                   // 현재 열

    // *********************************** 캐럿 ***********************************
    bool m_bCaretVisible;
    // *********************************** 캐럿 ***********************************

    afx_msg void OnFileSave2();
    afx_msg void OnFileOpen();
    afx_msg void OnFileNew();
    afx_msg void OnFileNewWindow();

    afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);

    afx_msg void OnChar(UINT nChar, UINT nRepCnt, UINT nFlags);

    // *********************************** 캐럿 ***********************************
    afx_msg void OnSetFocus(CWnd* pOldWnd);
    afx_msg void OnKillFocus(CWnd* pNewWnd);
    // *********************************** 캐럿 ***********************************

    // *********************************** 스크롤 ***********************************
    afx_msg void OnVScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
    afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
    // *********************************** 스크롤 ***********************************
};
