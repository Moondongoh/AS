
// MFC_PaintView.h: CMFCPaintView 클래스의 인터페이스
//

#pragma once

class CMFCPaintView : public CView
{
protected: // serialization에서만 만들어집니다.
    CMFCPaintView() noexcept;
    DECLARE_DYNCREATE(CMFCPaintView)

    // 특성입니다.
public:
    CMFCPaintDoc* GetDocument() const;

    // 작업입니다.
public:
    // 선 그리기 및 지우개 Part
    CPoint m_ptPrev; // 이전 점을 저장하기 위한 변수
    CPoint m_startPoint;
    CPoint m_endPoint;

    bool m_isDrawing; // 현재 그리고 있는지 여부
    bool m_isErasing; // 현재 지우개를 사용하고 있는지 여부
    bool m_isLine;
    bool m_isRect;
    bool m_isEllipse;

    bool isRed;
    bool isBlue;
    bool isGreen;


    bool F_isRed;
    bool F_isBlue;
    bool F_isGreen;
    // 선 굵기 Part
    int m_PenWidth; // 선 굵기

    // 선 색상
    COLORREF LineColor = 0x00000000;
    COLORREF FullColor = NULL_BRUSH;//NULL_BRUSH

    CBitmap m_BackBuffer;        // 더블 버퍼링을 위한 비트맵
    CDC m_BackBufferDC;          // 더블 버퍼링을 위한 DC

    // 재정의입니다.
public:
    virtual void OnDraw(CDC* pDC);  // 이 뷰를 그리기 위해 재정의되었습니다.
    virtual BOOL PreCreateWindow(CREATESTRUCT& cs);
protected:

    // 구현입니다.
public:
    virtual ~CMFCPaintView();
#ifdef _DEBUG
    virtual void AssertValid() const;
    virtual void Dump(CDumpContext& dc) const;
#endif

protected:

    // 생성된 메시지 맵 함수
protected:
    DECLARE_MESSAGE_MAP()
public:
    afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
    afx_msg void OnMouseMove(UINT nFlags, CPoint point);
    afx_msg void OnLButtonUp(UINT nFlags, CPoint point);
    afx_msg void OnRButtonDown(UINT nFlags, CPoint point);
    afx_msg void OnRButtonUp(UINT nFlags, CPoint point);
    afx_msg void OnLine1();
    afx_msg void OnLine10();
    afx_msg void OnLine15();
    afx_msg void OnRect();
    afx_msg void OnEllipse();
    afx_msg void OnLine();
    afx_msg void OnFileNew();
    afx_msg void OnFileOpen();
    afx_msg void OnFileSave();
    afx_msg void OnRed();
    afx_msg void OnBlue();
    afx_msg void OnGreen();
    afx_msg void OnLRed();
    afx_msg void OnLBlue();
    afx_msg void OnLGreen();
    afx_msg void OnVsave();
};
