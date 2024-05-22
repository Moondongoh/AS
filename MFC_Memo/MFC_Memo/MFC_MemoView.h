#pragma once
#define Row 1000
#define Columns 1000

class CMFCMemoView : public CScrollView
{
protected: // serialization에서만 만들어집니다.
    CMFCMemoView() noexcept;
    DECLARE_DYNCREATE(CMFCMemoView)

    // 특성입니다.
public:
    CMFCMemoDoc* GetDocument() const;

    // 재정의입니다.
public:
    virtual void OnDraw(CDC* pDC);  // 이 클래스에 대한 그리기입니다.
    virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
    // 인쇄 작업 준비를 위한 함수 선언
    virtual BOOL OnPreparePrinting(CPrintInfo* pInfo);
    // 인쇄 작업 시작을 위한 함수 선언
    virtual void OnBeginPrinting(CDC* pDC, CPrintInfo* pInfo);
    // 인쇄 작업 종료를 위한 함수 선언
    virtual void OnEndPrinting(CDC* pDC, CPrintInfo* pInfo);


    virtual void OnInitialUpdate();

protected:
    // 구현입니다.
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
    int m_nScrollWidth; // Width of the total drawing area
    int m_nScrollHeight; // Height of the total drawing area
    int m_nPageSize; // Size of a page for scrolling

protected:
    // 텍스트를 저장할 2차원 배열
    wchar_t m_textArray[Row][Columns]; // 예시로 1000x1000 배열 사용

    int m_nCurrentRow; // 현재 행
    int m_nCurrentColumn; // 현재 열
    // *********************************** 캐럿 ***********************************
    bool m_bCaretVisible;
    // *********************************** 캐럿 ***********************************

    // 키보드 이벤트 처리를 위한 핸들러
    afx_msg void OnChar(UINT nChar, UINT nRepCnt, UINT nFlags);

    DECLARE_MESSAGE_MAP()
public:
    afx_msg void OnFileSave2();
    afx_msg void OnFileOpen();
    afx_msg void OnFileNew();
    afx_msg void OnFileNewWindow();
    // *********************************** 캐럿 ***********************************
    afx_msg void OnSetFocus(CWnd* pOldWnd);
    afx_msg void OnKillFocus(CWnd* pNewWnd);
    // *********************************** 캐럿 ***********************************
    afx_msg void OnKeyDown(UINT nChar, UINT nRepCnt, UINT nFlags);
    afx_msg void OnVScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
    afx_msg void OnHScroll(UINT nSBCode, UINT nPos, CScrollBar* pScrollBar);
};
