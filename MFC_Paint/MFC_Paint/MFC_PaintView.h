
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
    bool m_bDrawing; // 현재 그리고 있는지 여부
    bool m_bErasing; // 현재 지우개를 사용하고 있는지 여부

    // 선 굵기 Part
    int m_nPenWidth; // 선 굵기

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
};
